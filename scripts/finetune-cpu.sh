#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PADDLE_MODEL_DIR="${HOME}/.paddlex/official_models/PP-OCRv6_medium_rec"
OCRAX_DIR="${REPO_ROOT}/../ocrax"

echo "=== Starting PP-OCRv6 Fine-Tuning on CPU ==="
ocrtorch-train \
  --checkpoint "${REPO_ROOT}/artifacts/ppocrv6_medium_rec.pt" \
  --inference-yaml "${PADDLE_MODEL_DIR}/inference.yml" \
  --train-manifest "${OCRAX_DIR}/data/manifests/train.jsonl" \
  --val-manifest "${OCRAX_DIR}/data/manifests/validation.jsonl" \
  --output-dir "${REPO_ROOT}/artifacts/checkpoints/finetuned_cpu" \
  --device cpu \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --patience 3 \
  "$@"
