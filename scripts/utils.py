import importlib.util
import importlib
import inspect
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import yaml

HF_CHECKPOINT_REPO_URL = "https://huggingface.co/opengait/OpenGait"
HF_CHECKPOINT_REPO_ID = "opengait/OpenGait"
OPENGAIT_CODE_REPO_URL = "https://github.com/ShiqiYu/OpenGait.git"
DATASET_NAMES = [
    "CASIA-B",
    "CCPG",
    "GREW",
    "Gait3D",
    "OUMVLP",
    "SUSTech1K",
]
CONFIG_META_CACHE: dict[Path, tuple[str, str, str]] = {}


@dataclass
class CheckpointInfo:
    checkpoint_path: Path
    dataset: str
    model_name: str
    experiment_dir: Path


def log(level: str, message: str, **fields: Any) -> None:
    payload = {"level": level.upper(), "message": message, **fields}
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def run_cmd(cmd: Iterable[str], cwd: Optional[Path] = None) -> None:
    cmd_list = list(cmd)
    log("info", "running_command", command=" ".join(cmd_list), cwd=str(cwd or Path.cwd()))
    process = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            text = line.rstrip("\n")
            if text:
                log("debug", "command_output", line=text)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}): {' '.join(cmd_list)}")


def ensure_git_repo(repo_root: Path, remote_url: str, label: str) -> None:
    git_dir = repo_root / ".git"
    if git_dir.exists():
        log("info", f"{label}_exists", path=str(repo_root))
        return

    if repo_root.exists() and any(repo_root.iterdir()):
        raise RuntimeError(f"{label} path exists but is not a git repository: {repo_root}")

    repo_root.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", remote_url, str(repo_root)])
    log("info", f"{label}_cloned", path=str(repo_root), source=remote_url)


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def ensure_lfs_materialized(repo_root: Path) -> None:
    pt_files = sorted(p for p in repo_root.rglob("*.pt") if ".git" not in p.parts)
    if not pt_files:
        log("warning", "no_pt_files_found", repo_root=str(repo_root))
        return

    pointer_before = [p for p in pt_files if is_lfs_pointer(p)]
    if not pointer_before:
        log("info", "lfs_materialized", checked=len(pt_files))
        return

    log("info", "lfs_pull_required", total=len(pt_files), pointer_files=len(pointer_before))
    run_cmd(["git", "lfs", "install", "--local"], cwd=repo_root)
    try:
        run_cmd(["git", "lfs", "pull"], cwd=repo_root)
    except Exception as exc:
        # Some upstream pointers are missing on Hugging Face LFS (404).
        # Continue so available checkpoints can still be exported.
        log("warning", "lfs_pull_partial_failure", error=str(exc))

    pointer_after = [p for p in pt_files if is_lfs_pointer(p)]
    if pointer_after:
        pointer_after = recover_pointer_files_with_hf_hub(repo_root, pointer_after)

    if pointer_after:
        log(
            "warning",
            "lfs_unavailable_checkpoints",
            count=len(pointer_after),
            examples=[str(p.relative_to(repo_root)) for p in pointer_after[:5]],
        )
        return

    log("info", "lfs_pull_completed", checked=len(pt_files))


def recover_pointer_files_with_hf_hub(repo_root: Path, pointer_files: list[Path]) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        log("warning", "lfs_hf_hub_unavailable", error=str(exc))
        return pointer_files

    recovered = 0
    failures: list[tuple[Path, str]] = []

    for path in pointer_files:
        rel = path.relative_to(repo_root).as_posix()
        try:
            cached = hf_hub_download(
                repo_id=HF_CHECKPOINT_REPO_ID,
                filename=rel,
                repo_type="model",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, path)
            recovered += 1
        except Exception as exc:
            failures.append((path, str(exc)))

    if recovered:
        log("info", "lfs_hf_hub_recovered", recovered=recovered)

    if failures:
        log(
            "warning",
            "lfs_hf_hub_recovery_failed",
            failed=len(failures),
            examples=[str(p.relative_to(repo_root)) for p, _ in failures[:5]],
        )

    return [p for p in pointer_files if is_lfs_pointer(p)]


def ensure_checkpoint_repo(repo_root: Path) -> None:
    ensure_git_repo(repo_root, HF_CHECKPOINT_REPO_URL, "checkpoint_repo")
    ensure_lfs_materialized(repo_root)


def ensure_code_repo(code_root: Path) -> None:
    ensure_git_repo(code_root, OPENGAIT_CODE_REPO_URL, "code_repo")


def ensure_output_dirs(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_NAMES:
        (output_root / dataset).mkdir(parents=True, exist_ok=True)


def discover_checkpoints(repo_root: Path) -> list[CheckpointInfo]:
    checkpoints = sorted(p for p in repo_root.rglob("*.pt") if ".git" not in p.parts)
    materialized = [p for p in checkpoints if not is_lfs_pointer(p)]
    skipped = len(checkpoints) - len(materialized)
    if skipped:
        log("warning", "checkpoint_skip_unavailable_lfs", skipped=skipped)

    items: list[CheckpointInfo] = []
    for checkpoint_path in materialized:
        info = parse_checkpoint_info(repo_root, checkpoint_path)
        items.append(info)

    log("info", "checkpoint_discovery_complete", count=len(items))
    return items


def parse_checkpoint_info(repo_root: Path, checkpoint_path: Path) -> CheckpointInfo:
    rel = checkpoint_path.relative_to(repo_root)
    parts = rel.parts
    dataset = parts[0] if parts else "unknown"

    if checkpoint_path.parent.name == "checkpoints":
        experiment_dir = checkpoint_path.parent.parent
    else:
        experiment_dir = checkpoint_path.parent

    model_name = experiment_dir.name
    return CheckpointInfo(
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        model_name=model_name,
        experiment_dir=experiment_dir,
    )


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def config_metadata(path: Path) -> tuple[str, str, str]:
    cached = CONFIG_META_CACHE.get(path)
    if cached is not None:
        return cached

    model_name = ""
    dataset_name = ""
    save_name = ""
    try:
        data = load_yaml(path)
        model_name = str((data.get("model_cfg") or {}).get("model", ""))
        dataset_name = str((data.get("data_cfg") or {}).get("dataset_name", ""))
        save_name = str((data.get("evaluator_cfg") or {}).get("save_name", ""))
    except Exception:
        pass

    meta = (normalize_name(model_name), normalize_name(dataset_name), normalize_name(save_name))
    CONFIG_META_CACHE[path] = meta
    return meta


def infer_expected_models(info: CheckpointInfo) -> set[str]:
    key = normalize_name("/".join(info.checkpoint_path.parts) + info.model_name)
    model_names: set[str] = set()

    if "swingait" in key:
        model_names.add("SwinGait")
    if "gaitgraph1" in key:
        model_names.add("GaitGraph1")
    if "gaitgraph2" in key:
        model_names.add("GaitGraph2")
    if "gaittr" in key:
        model_names.add("GaitTR")
    if "skeletongaitpp" in key:
        model_names.add("SkeletonGaitPP")
    if "skeletongait" in key and "skeletongaitpp" not in key:
        model_names.add("DeepGaitV2")
    if "deepgaitv2" in key:
        model_names.add("DeepGaitV2")
    if "gaitbase" in key or "baseline" in key:
        model_names.add("Baseline")
    if "gaitset" in key:
        model_names.add("GaitSet")
    if "gaitpart" in key:
        model_names.add("GaitPart")
    if "gaitgl" in key:
        model_names.add("GaitGL")
    if "gln" in key:
        model_names.add("GLN")
    if "parsinggait" in key:
        model_names.add("ParsingGait")
    if "smplgait" in key:
        model_names.add("SMPLGait")
    if "biggait" in key and "biggergait" not in key:
        model_names.add("BigGait__Dinov2_Gaitbase")
    if "biggergait" in key:
        model_names.add("BiggerGait__DINOv2")
    if "multigait" in key or "mtsgait" in key:
        model_names.add("MultiGaitpp")

    return {normalize_name(name) for name in model_names}


def find_config_for_checkpoint(repo_root: Path, code_root: Path, info: CheckpointInfo) -> Optional[Path]:
    candidates: list[Path] = []
    bases = [
        info.experiment_dir,
        info.experiment_dir.parent,
        repo_root / "configs",
        code_root / "configs",
    ]

    for base in bases:
        if not base.exists():
            continue
        candidates.extend(sorted(base.rglob("*.yaml")))
        candidates.extend(sorted(base.rglob("*.yml")))

    if not candidates:
        return None

    stem_lower = info.checkpoint_path.stem.lower()
    model_key = normalize_name(info.model_name)
    dataset_key = normalize_name(info.dataset)
    ckpt_key = normalize_name(stem_lower.split("-")[0])
    expected_models = infer_expected_models(info)
    expects_17 = "_17" in stem_lower or "_17" in info.model_name.lower()

    def score(path: Path) -> Tuple[int, int]:
        p = str(path).lower()
        pkey = normalize_name(p)
        cfg_model, cfg_dataset, cfg_save = config_metadata(path)
        has_17 = "_17" in path.name.lower()

        value = 0
        if cfg_dataset and dataset_key and cfg_dataset == dataset_key:
            value += 18
        elif dataset_key and dataset_key in pkey:
            value += 3

        if expected_models and cfg_model in expected_models:
            value += 30
        elif expected_models and cfg_model:
            value -= 4

        if cfg_save and (model_key in cfg_save or ckpt_key in cfg_save):
            value += 8

        if model_key and model_key in pkey:
            value += 6
        if ckpt_key and ckpt_key in pkey:
            value += 8

        if expects_17 == has_17:
            value += 2
        else:
            value -= 2

        if "configs" in path.parts:
            value += 1

        return (value, -len(path.parts))

    best = sorted(candidates, key=score, reverse=True)[0]
    return best


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a dictionary: {path}")
    return data


def load_builder_from_main_file(main_py: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {main_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_model", None)
    if builder is None:
        raise ImportError(f"'build_model' not found in {main_py}")
    return builder


class _NoOpMsgMgr:
    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop


class _HiddenStateOut:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _BiggerGaitBackboneProxy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from modeling.models.BigGait_utils.DINOv2 import vit_small

        self.vit = vit_small(logger=_NoOpMsgMgr())

    def forward(self, x, output_hidden_states=True):
        tokens = self.vit.prepare_tokens_with_masks(x)
        hidden_states = [tokens]
        for blk in self.vit.blocks:
            tokens = blk(tokens)
            hidden_states.append(tokens)
        return _HiddenStateOut(tuple(hidden_states))


def patch_cpu_unsafe_ops() -> None:
    try:
        from modeling.backbones import resgcn as resgcn_mod

        if not getattr(resgcn_mod.ResGCNModule, "_opengait_export_cpu_patch", False):
            def _resgcn_forward(self, x, A):
                A = A.to(device=x.device)
                return self.tcn(self.scn(x, A * self.edge), self.residual(x))

            resgcn_mod.ResGCNModule.forward = _resgcn_forward
            setattr(resgcn_mod.ResGCNModule, "_opengait_export_cpu_patch", True)
    except Exception:
        pass

    try:
        from modeling.models import gaittr as gaittr_mod

        if not getattr(gaittr_mod.STModule, "_opengait_export_cpu_patch", False):
            def _stmodule_forward(self, x):
                n, c, t, v = x.size()
                x = x.permute(0, 1, 3, 2).reshape(n, c * v, t)
                x = self.data_bn(x)
                x = x.reshape(n, c, v, t).permute(0, 1, 3, 2)
                self.incidence = self.incidence.to(device=x.device)
                xa = x.permute(0, 2, 1, 3).reshape(-1, c, 1, v)
                attn_out = self.attention_conv(xa)
                attn_out = attn_out.reshape(n, t, -1, v).permute(0, 2, 1, 3)
                y = attn_out
                y = self.bn(self.relu(y))
                return y

            gaittr_mod.STModule.forward = _stmodule_forward
            setattr(gaittr_mod.STModule, "_opengait_export_cpu_patch", True)
    except Exception:
        pass


def make_lightweight_opengait_builder():
    import modeling.base_model as base_model
    from modeling import models

    patch_cpu_unsafe_ops()

    def lightweight_base_model_init(self, cfgs: Dict[str, Any], training: bool):
        torch.nn.Module.__init__(self)
        self.msg_mgr = _NoOpMsgMgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs.get("trainer_cfg", {}) if training else cfgs.get("evaluator_cfg", {})

        model_cfg = cfgs.get("model_cfg")
        if not isinstance(model_cfg, dict):
            raise ValueError("Config missing model_cfg dictionary")

        self.build_network(model_cfg)
        self.train(training)

    def detect_adapter(model: torch.nn.Module) -> tuple[str, int]:
        name = model.__class__.__name__
        if name == "GaitGraph1":
            return ("tuple_gaitgraph1", 3)
        if name == "GaitGraph2":
            return ("tuple_gaitgraph2", 5)
        if name == "GaitTR":
            return ("tuple_gaittr", 10)
        if name == "SMPLGait":
            return ("tuple_smplgait", 1)
        if name in {"GaitSet", "GaitPart", "GaitGL", "GLN", "ParsingGait", "SwinGait"}:
            return ("tuple_sequence_silhouette", 1)
        if name == "SkeletonGaitPP":
            return ("tuple_sequence_channels", 3)
        if name == "MultiGaitpp":
            part1 = int(getattr(model, "part1", 2))
            part2 = int(getattr(model, "part2", 1))
            return ("tuple_sequence_channels", max(1, part1 + part2))
        if name == "BigGait__Dinov2_Gaitbase":
            return ("tuple_biggait", 3)
        if name == "BiggerGait__DINOv2":
            return ("tuple_sequence_channels", 3)
        if name == "DeepGaitV2":
            try:
                layer0 = getattr(model, "layer0")
                conv = layer0.forward_block[0]  # type: ignore[attr-defined]
                in_channels = int(getattr(conv, "in_channels", 1))
            except Exception:
                in_channels = 1
            if in_channels == 1:
                return ("tuple_sequence_silhouette", 1)
            return ("tuple_sequence_channels", max(1, in_channels))
        if name == "Baseline":
            try:
                layer0 = next(m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)))
                in_channels = int(getattr(layer0, "in_channels", 1))
            except Exception:
                in_channels = 1
            if in_channels == 1:
                return ("tuple_sequence_silhouette", 1)
            return ("tuple_sequence_channels", max(1, in_channels))
        return ("tuple_sequence_channels", 1)

    def builder(cfgs: Dict[str, Any], training: bool = False):
        model_cfg = cfgs.get("model_cfg")
        if not isinstance(model_cfg, dict):
            raise ValueError("Config missing model_cfg dictionary")

        model_name = model_cfg.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Config missing model_cfg.model")

        Model = getattr(models, model_name, None)
        if Model is None:
            raise ValueError(f"Model class not found in OpenGait registry: {model_name}")

        original_init = base_model.BaseModel.__init__
        base_model.BaseModel.__init__ = lightweight_base_model_init
        try:
            model = Model(cfgs, training)
        finally:
            base_model.BaseModel.__init__ = original_init

        model_name = model.__class__.__name__
        if model_name == "BigGait__Dinov2_Gaitbase" and not hasattr(model, "backbone"):
            try:
                from modeling.models.BigGait_utils.DINOv2 import vit_small

                model.backbone = vit_small(logger=_NoOpMsgMgr())
            except Exception:
                pass
        if model_name == "BiggerGait__DINOv2" and not hasattr(model, "Backbone"):
            try:
                model.Backbone = _BiggerGaitBackboneProxy()
            except Exception:
                pass

        adapter, channels = detect_adapter(model)
        setattr(model, "_opengait_export_adapter", adapter)
        setattr(model, "_opengait_export_channels", channels)
        if model.__class__.__name__ == "GaitGraph1":
            # The OpenGait implementation has an eval-path bug when tta=False.
            setattr(model, "tta", True)
        return model

    return builder


def import_build_model(code_root: Path):
    code_root = code_root.resolve()

    for path in [str(code_root), str(code_root / "opengait")]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # OpenGait imports top-level `utils`; remove this script module alias if present.
    loaded_utils = sys.modules.get("utils")
    this_file = Path(__file__).resolve()
    loaded_utils_file = Path(getattr(loaded_utils, "__file__", "")) if loaded_utils else None
    if loaded_utils_file and loaded_utils_file.resolve() == this_file:
        del sys.modules["utils"]

    # Prime OpenGait namespaces to avoid collisions with script-level modules.
    for module_name in ["utils", "data", "evaluation", "modeling"]:
        try:
            importlib.import_module(module_name)
        except Exception:
            # Not all namespaces are required for every model import path.
            pass

    first_error = ""
    try:
        from opengait.main import build_model as builder  # type: ignore

        log("info", "builder_imported", module="opengait.main")
        return builder
    except Exception as first_exc:
        first_error = str(first_exc)
        log("warning", "builder_import_failed", module="opengait.main", error=first_error)

    second_error = ""
    candidate_main_files = [code_root / "main.py", code_root / "opengait" / "main.py"]
    candidate_main_files.extend(sorted(code_root.rglob("main.py")))

    seen: set[Path] = set()
    for idx, main_py in enumerate(candidate_main_files):
        if main_py in seen or not main_py.exists():
            continue
        seen.add(main_py)

        try:
            with main_py.open("r", encoding="utf-8", errors="ignore") as f:
                contents = f.read()
            if "def build_model" not in contents:
                continue

            module_name = f"opengait_repo_main_{idx}"
            builder = load_builder_from_main_file(main_py, module_name)
            log("info", "builder_imported", module=str(main_py))
            return builder
        except Exception as third_exc:
            second_error = str(third_exc)

    if not second_error:
        second_error = "No main.py file with build_model was found under code root"

    third_error = ""
    try:
        builder = make_lightweight_opengait_builder()
        log("info", "builder_imported", module="modeling.models (lightweight_fallback)")
        return builder
    except Exception as third_exc:
        third_error = str(third_exc)

    raise ImportError(
        "Unable to import build_model from opengait.main, file-based main.py, or lightweight modeling fallback. "
        f"Errors: {first_error}; {second_error}; {third_error}"
    )


def try_build_model(builder, config: Dict[str, Any]):
    candidate_kwargs = [
        {},
        {"cfgs": config},
        {"config": config},
        {"training": False},
        {"is_train": False},
        {"cfgs": config, "training": False},
        {"cfgs": config, "is_train": False},
    ]

    signature = inspect.signature(builder)
    accepted = set(signature.parameters.keys())

    errors: list[str] = []
    for kwargs in candidate_kwargs:
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        try:
            if not accepted:
                return builder()
            if "cfgs" in accepted and "cfgs" not in filtered:
                filtered["cfgs"] = config
            return builder(**filtered)
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError(f"Unable to build model with available signatures: {errors}")


def checkpoint_to_state_dict(raw_checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw_checkpoint, torch.nn.Module):
        return raw_checkpoint.state_dict()

    if isinstance(raw_checkpoint, dict):
        for key in [
            "model",
            "state_dict",
            "model_state_dict",
            "model_state",
            "net",
            "encoder",
        ]:
            if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
                return raw_checkpoint[key]

        if all(isinstance(k, str) for k in raw_checkpoint.keys()):
            return raw_checkpoint

    raise ValueError("Unsupported checkpoint format")


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def load_torch_checkpoint(checkpoint_path: Path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = load_torch_checkpoint(checkpoint_path)
    state_dict = checkpoint_to_state_dict(checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)
    model_state = model.state_dict()
    filtered_state = {}
    mismatched_shapes = 0
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(value.shape) != tuple(target.shape):
            mismatched_shapes += 1
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    log(
        "info",
        "checkpoint_loaded",
        checkpoint=str(checkpoint_path),
        loaded_keys=len(filtered_state),
        shape_mismatches=mismatched_shapes,
        missing_keys=len(missing),
        unexpected_keys=len(unexpected),
    )


def find_embedding_output(model_output: Any):
    def search(value: Any) -> Optional[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, dict):
            priority_keys = [
                "embedding",
                "embeddings",
                "feat",
                "feature",
                "output",
                "inference_feat",
            ]
            for key in priority_keys:
                if key not in value:
                    continue
                found = search(value[key])
                if found is not None:
                    return found

            for child in value.values():
                found = search(child)
                if found is not None:
                    return found

        if isinstance(value, (list, tuple)):
            for child in value:
                found = search(child)
                if found is not None:
                    return found

        return None

    found = search(model_output)
    if found is None:
        raise ValueError("Unable to extract tensor output for ONNX export")
    return found


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def _slice_or_pad_channels(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        if channels <= x.size(2):
            return x[:, :, :channels, :, :]
        pad = channels - x.size(2)
        zeros = torch.zeros((x.size(0), x.size(1), pad, x.size(3), x.size(4)), device=x.device, dtype=x.dtype)
        return torch.cat([x, zeros], dim=2)

    def _as_opengait_sequence_tuple_input(self, x: torch.Tensor, channels: int):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [N, T, C, H, W], got shape {tuple(x.shape)}")

        n, s, _, h, w = x.shape
        sequence = self._slice_or_pad_channels(x, channels)
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        types = torch.zeros((n,), dtype=torch.long, device=x.device)
        views = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), s, dtype=torch.long, device=x.device)
        return ([sequence], labels, types, views, seq_lens)

    def _as_opengait_silhouette_tuple_input(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [N, T, C, H, W], got shape {tuple(x.shape)}")

        n, s, _, _, _ = x.shape
        sequence = x[:, :, 0, :, :].contiguous()
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        types = torch.zeros((n,), dtype=torch.long, device=x.device)
        views = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), s, dtype=torch.long, device=x.device)
        return ([sequence], labels, types, views, seq_lens)

    def _as_biggait_tuple_input(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [N, T, C, H, W], got shape {tuple(x.shape)}")
        n, s, _, h, w = x.shape
        sequence = self._slice_or_pad_channels(x, 3)
        ratio = torch.full((n, s, 1), float(w) / float(h), dtype=x.dtype, device=x.device)
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        types = torch.zeros((n,), dtype=torch.long, device=x.device)
        views = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), s, dtype=torch.long, device=x.device)
        return ([sequence, ratio], labels, types, views, seq_lens)

    def _as_gaitgraph1_tuple_input(self, x: torch.Tensor):
        n, t, _, h, _ = x.shape
        v = int(getattr(getattr(self.model, "graph", None), "num_node", 17))
        c = int(getattr(self.model, "input_branch", [3])[0])
        i = int(getattr(self.model, "input_num", 1))
        base = x[:, :, 0, :, :].mean(dim=-1)
        base = base[:, :, :v]
        pose = base.unsqueeze(1).unsqueeze(-1).repeat(1, c, 1, 1, i)
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        dummy = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), t, dtype=torch.long, device=x.device)
        return ([pose], labels, dummy, dummy, seq_lens)

    def _as_gaitgraph2_tuple_input(self, x: torch.Tensor):
        n, t, _, _, _ = x.shape
        v = int(getattr(getattr(self.model, "graph", None), "num_node", 17))
        i = int(getattr(self.model, "input_num", 3))
        c = int(getattr(self.model, "input_branch", [5])[0])
        base = x[:, :, 0, :, :].mean(dim=-1)
        base = base[:, :, :v]
        pose = base.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, i, c)
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        dummy = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), t, dtype=torch.long, device=x.device)
        return ([pose], labels, dummy, dummy, seq_lens)

    def _as_gaittr_tuple_input(self, x: torch.Tensor):
        n, t, _, _, _ = x.shape
        v = int(getattr(getattr(self.model, "graph", None), "num_node", 17))
        c = int(getattr(self.model, "in_channels", [10])[0])
        m = 1
        base = x[:, :, 0, :, :].mean(dim=-1)
        base = base[:, :, :v]
        pose = base.unsqueeze(2).unsqueeze(-1).repeat(1, 1, c, 1, m)
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        dummy = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), t, dtype=torch.long, device=x.device)
        return ([pose], labels, dummy, dummy, seq_lens)

    def _as_smplgait_tuple_input(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [N, T, C, H, W], got shape {tuple(x.shape)}")
        n, s, _, _, _ = x.shape
        sils = x[:, :, 0, :, :].contiguous()
        smpls = torch.zeros((n, s, 85), dtype=x.dtype, device=x.device)
        labels = torch.zeros((n,), dtype=torch.long, device=x.device)
        dummy = torch.zeros((n,), dtype=torch.long, device=x.device)
        seq_lens = torch.full((1, n), s, dtype=torch.long, device=x.device)
        return ([sils, smpls], labels, dummy, dummy, seq_lens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter = getattr(self.model, "_opengait_export_adapter", "")
        channels = int(getattr(self.model, "_opengait_export_channels", 1))
        if adapter == "tuple_sequence_channels":
            out = self.model(self._as_opengait_sequence_tuple_input(x, channels))
        elif adapter == "tuple_sequence_silhouette":
            out = self.model(self._as_opengait_silhouette_tuple_input(x))
        elif adapter == "tuple_biggait":
            out = self.model(self._as_biggait_tuple_input(x))
        elif adapter == "tuple_gaitgraph1":
            out = self.model(self._as_gaitgraph1_tuple_input(x))
        elif adapter == "tuple_gaitgraph2":
            out = self.model(self._as_gaitgraph2_tuple_input(x))
        elif adapter == "tuple_gaittr":
            out = self.model(self._as_gaittr_tuple_input(x))
        elif adapter == "tuple_smplgait":
            out = self.model(self._as_smplgait_tuple_input(x))
        else:
            out = self.model(x)
        return find_embedding_output(out)
