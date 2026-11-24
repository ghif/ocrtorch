#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PADDLE_MODEL_DIR="${HOME}/.paddlex/official_models/PP-OCRv6_medium_rec"

echo "=== Starting PP-OCRv6 Hybrid bf16 Fine-Tuning on Apple Silicon MPS ==="
ocrtorch-train \
  --config "${REPO_ROOT}/configs/local_mps.yaml" \
  --checkpoint "${REPO_ROOT}/artifacts/ppocrv6_medium_rec.pt" \
  --inference-yaml "${PADDLE_MODEL_DIR}/inference.yml" \
  --train-manifest "${REPO_ROOT}/data/manifests/train.jsonl" \
  --val-manifest "${REPO_ROOT}/data/manifests/validation.jsonl" \
  --output-dir "${REPO_ROOT}/artifacts/checkpoints/finetuned_mps_hybrid_bs64" \
  "$@"
