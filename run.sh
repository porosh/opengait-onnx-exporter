#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Error: docker compose is not available."
  echo "Install Docker Desktop (or docker-compose) and try again."
  exit 1
fi

echo "Building exporter image..."
"${COMPOSE_CMD[@]}" build exporter

echo "Running ONNX export..."
"${COMPOSE_CMD[@]}" run --rm exporter

echo "Done. ONNX outputs are under: $ROOT_DIR/output"
