import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

from utils import (
    OnnxExportWrapper,
    discover_checkpoints,
    ensure_checkpoint_repo,
    ensure_code_repo,
    ensure_output_dirs,
    find_config_for_checkpoint,
    import_build_model,
    load_checkpoint_weights,
    load_torch_checkpoint,
    load_yaml,
    log,
    parse_checkpoint_info,
    try_build_model,
)

# OpenGait silhouette models generally operate on 64x44 inputs.
DEFAULT_INPUT_SHAPE = (1, 30, 3, 64, 44)
DYNAMIC_AXES = {
    "input": {0: "batch_size", 1: "frame_count"},
    "embedding": {0: "batch_size"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all OpenGait checkpoints to ONNX")
    parser.add_argument("--repo-root", type=Path, default=Path("repo"))
    parser.add_argument("--code-root", type=Path, default=Path("opengait_code"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--opset", type=int, default=16)
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint_path: Path,
    repo_root: Path,
    code_root: Path,
    builder,
) -> torch.nn.Module:
    info = parse_checkpoint_info(repo_root, checkpoint_path)
    config_path = find_config_for_checkpoint(repo_root, code_root, info)

    if config_path is not None:
        cfg = load_yaml(config_path)
        model_cfg = cfg.get("model_cfg")
        if isinstance(model_cfg, dict) and model_cfg.get("model") == "MultiGaitpp":
            backbone_cfg = model_cfg.get("Backbone")
            if isinstance(backbone_cfg, dict) and "in_channels" not in backbone_cfg:
                # Some upstream MultiGait++ configs omit this unused field, but
                # the model constructor still accesses it.
                part1 = int(backbone_cfg.get("part1_channel", 1))
                part2 = int(backbone_cfg.get("part2_channel", 1))
                backbone_cfg["in_channels"] = part1 + part2
        log("info", "config_selected", checkpoint=str(checkpoint_path), config=str(config_path))
        model = try_build_model(builder, cfg)
        load_checkpoint_weights(model, checkpoint_path)
        model.eval()
        return model

    checkpoint = load_torch_checkpoint(checkpoint_path)
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        log("warning", "config_missing_using_checkpoint_module", checkpoint=str(checkpoint_path))
        return checkpoint

    raise RuntimeError(f"No usable config found and checkpoint is not a torch.nn.Module: {checkpoint_path}")


def export_onnx(
    checkpoint_path: Path,
    repo_root: Path,
    code_root: Path,
    output_root: Path,
    builder,
    opset: int,
) -> Path:
    dataset = checkpoint_path.relative_to(repo_root).parts[0]
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_checkpoint(checkpoint_path, repo_root, code_root, builder)
    wrapped = OnnxExportWrapper(model).eval()

    dummy_input = torch.randn(*DEFAULT_INPUT_SHAPE, dtype=torch.float32)
    onnx_path = output_dir / f"{checkpoint_path.stem}.onnx"

    export_kwargs = dict(
        export_params=True,
        dynamo=False,
        opset_version=opset,
        input_names=["input"],
        output_names=["embedding"],
    )
    try:
        torch.onnx.export(
            wrapped,
            dummy_input,
            str(onnx_path),
            dynamic_axes=DYNAMIC_AXES,
            **export_kwargs,
        )
    except Exception as exc:
        text = str(exc)
        # Some OpenGait models build pooling kernel sizes from runtime tensor shapes.
        # Retry once with static axes so those kernel sizes become constant.
        if "Failed to export a node" not in text and "not constant" not in text:
            raise
        log(
            "warning",
            "export_retry_static_axes",
            checkpoint=str(checkpoint_path),
            reason="symbolic_requires_constant_shape",
        )
        torch.onnx.export(
            wrapped,
            dummy_input,
            str(onnx_path),
            dynamic_axes=None,
            **export_kwargs,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _ = session.run(
        None,
        {
            "input": dummy_input.cpu().numpy(),
        },
    )

    return onnx_path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root
    code_root = args.code_root
    output_root = args.output_root

    try:
        ensure_output_dirs(output_root)
        ensure_checkpoint_repo(repo_root)
        ensure_code_repo(code_root)
    except Exception as exc:
        log("error", "bootstrap_failed", error=str(exc))
        return 2

    try:
        builder = import_build_model(code_root)
    except Exception as exc:
        log("error", "builder_unavailable", error=str(exc))
        return 2

    checkpoints = [item.checkpoint_path for item in discover_checkpoints(repo_root)]
    if not checkpoints:
        log("warning", "no_checkpoints_found", repo_root=str(repo_root))
        return 0

    success = 0
    failed = 0
    for checkpoint_path in checkpoints:
        log("info", "export_started", checkpoint=str(checkpoint_path))
        try:
            onnx_path = export_onnx(
                checkpoint_path=checkpoint_path,
                repo_root=repo_root,
                code_root=code_root,
                output_root=output_root,
                builder=builder,
                opset=args.opset,
            )
            success += 1
            log("info", "export_success", checkpoint=str(checkpoint_path), onnx=str(onnx_path))
        except Exception as exc:
            failed += 1
            log("error", "export_failed", checkpoint=str(checkpoint_path), error=str(exc))

    log("info", "export_summary", total=len(checkpoints), success=success, failed=failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
