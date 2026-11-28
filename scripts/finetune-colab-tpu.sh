#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_activate_env.sh"
activate_ocrtorch_env "${REPO_ROOT}" || true

echo "=== Launching PP-OCRv6 Hybrid bf16 Fine-Tuning with TorchTPU on Colab CLI ==="
python -m ocrtorch.cli.colab \
  --config "${REPO_ROOT}/configs/tpu_v6e.yaml" \
  --model-gcs gs://handnotes-ocr/model-dir \
  --dataset-gcs gs://all-datasets/RxHandBD \
  --tpu v5e1 \
  "$@"
