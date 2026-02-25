# OpenGait ONNX Exporter

Automated exporter for converting OpenGait checkpoints (`.pt`) into ONNX models, with per-model validation.

## What It Does

- Ensures checkpoint repo exists at `repo/` (`https://huggingface.co/opengait/OpenGait`).
- Ensures OpenGait source exists at `opengait_code/` (`https://github.com/ShiqiYu/OpenGait.git`).
- Recursively finds `checkpoints/*.pt`.
- Selects model config and builds model with fallback import paths.
- Loads checkpoint weights (shape-safe partial loading when required).
- Exports ONNX files into `output/<dataset>/`.
- Validates each ONNX using:
  - `onnx.checker`
  - `onnxruntime` CPU inference smoke test
- Continues on failures and prints an export summary.

## Environment

- Host: macOS Apple Silicon (M2 tested)
- Runtime: Docker (ARM64)
- Base image: `python:3.10-slim`

## Quick Start

Run full export in Docker:

```bash
./run.sh
```

Or run directly (if dependencies are already installed):

```bash
python scripts/export_all.py
```

## Export Defaults

From `scripts/export_all.py`:

- Dummy input shape: `(1, 30, 3, 64, 44)` as `[N, T, C, H, W]`
- Opset: `16`
- Dynamic axes:
  - `input`: batch (`0`), frames (`1`)
  - `embedding`: batch (`0`)

For models requiring static-shape export, the exporter retries automatically without dynamic axes.

## Verify All Generated ONNX

You can re-verify every ONNX under `output/`:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

files = sorted(Path("output").rglob("*.onnx"))
print(f"Checking {len(files)} files")

for p in files:
    m = onnx.load(str(p))
    onnx.checker.check_model(m)
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    feed = {}
    for i in s.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if len(shape) == 5:
            shape = [1, 30, 3, 64, 44]
        feed[i.name] = np.random.randn(*shape).astype(np.float32)
    _ = s.run(None, feed)
    print("OK", p)
PY
```

## Project Layout

- `scripts/export_all.py`: main export entrypoint
- `scripts/utils.py`: setup/discovery/build/load/export helpers
- `output/`: exported ONNX files (grouped by dataset)
- `repo/`: OpenGait checkpoint repository
- `opengait_code/`: OpenGait source tree used for model construction

## Git / Large Files

ONNX files are large; use Git LFS:

```bash
git lfs install
git lfs track "*.onnx"
```

This repository is configured with:

```gitattributes
*.onnx filter=lfs diff=lfs merge=lfs -text
```
