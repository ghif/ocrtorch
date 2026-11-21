#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Launching PP-OCRv6 Fine-Tuning with TorchTPU on Colab CLI (TPU v6e) ==="
python -m ocrtorch.cli.colab \
  --model-gcs gs://handnotes-ocr/model-dir \
  --dataset-gcs gs://all-datasets/RxHandBD \
  --tpu tpuv6e \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --patience 3 \
  "$@"
