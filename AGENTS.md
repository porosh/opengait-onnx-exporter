# OpenGait ONNX Exporter

This repository exports all OpenGait `.pt` checkpoints to ONNX in a fully automated flow.

## Target Environment

- Host: Mac M2 (Apple Silicon)
- Runtime: Docker (ARM64 compatible)
- Base image: `python:3.10-slim`

## What This Project Does

- Clones `https://huggingface.co/opengait/OpenGait` into `repo/` if missing.
- Pulls checkpoint binaries with `git lfs pull` (so `.pt` files are real weights, not LFS pointers).
- Clones OpenGait source code from `https://github.com/ShiqiYu/OpenGait.git` into `opengait_code/`.
- Recursively discovers `checkpoints/*.pt` files.
- Builds models through OpenGait builder import fallback chain:
  - `from opengait.main import build_model`
  - file-based load from `main.py` in `opengait_code/`
- Loads checkpoint weights and exports ONNX with dynamic batch/frame axes.
- Validates each ONNX via `onnx.checker` and `onnxruntime`.
- Logs failures per checkpoint and continues processing.

## ONNX Export Defaults

- Input shape: `(1, 30, 3, 256, 128)` as `[batch, frames, channels, height, width]`
- Dynamic axes:
  - `input`: batch and frame dimensions
  - `embedding`: batch dimension
- Opset: `16`

## Directory Layout

- `scripts/export_all.py`: Main export pipeline.
- `scripts/utils.py`: Shared utilities for repo setup, discovery, loading, logging.
- `output/`: Export target directory, grouped by dataset.
- `repo/`: Hugging Face checkpoint repository.
- `opengait_code/`: OpenGait source code repository.

## Run Locally (inside Docker)

```bash
./run.sh
```

## Run Manually in Container

```bash
python scripts/export_all.py
```
