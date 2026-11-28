#!/usr/bin/env bash
set -euo pipefail

# Run PP-OCRv6 fine-tuning on a provisioned Google Cloud TPU VM (e.g. v6e) using TorchTPU.
# Prerequisite: run ./scripts/setup-tpu-vm.sh once on the VM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
CONDA_ENV="${CONDA_ENV:-torch-tpu}"
TPU_DATA_DIR="${TPU_DATA_DIR:-${REPO_ROOT}/data/tpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/checkpoints/finetuned_tpu_v6e}"
OUTPUT_GCS="${OUTPUT_GCS:-}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_activate_env.sh"
activate_ocrtorch_env "${REPO_ROOT}" || true

CHECKPOINT_PATH="${TPU_DATA_DIR}/model-dir/ppocrv6_medium_rec.pt"
INFERENCE_YAML="${TPU_DATA_DIR}/model-dir/inference.yml"
TRAIN_MANIFEST="${TPU_DATA_DIR}/manifests/train.jsonl"
VAL_MANIFEST="${TPU_DATA_DIR}/manifests/validation.jsonl"

for required in "${CHECKPOINT_PATH}" "${INFERENCE_YAML}" "${TRAIN_MANIFEST}" "${VAL_MANIFEST}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    echo "Run ./scripts/setup-tpu-vm.sh first (or set TPU_DATA_DIR)." >&2
    exit 1
  fi
done

echo "=== Starting PP-OCRv6 Fine-Tuning on TPU VM with TorchTPU ==="
ocrtorch-train \
  --config "${REPO_ROOT}/configs/tpu_v6e.yaml" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --inference-yaml "${INFERENCE_YAML}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

if [[ -n "${OUTPUT_GCS}" ]]; then
  echo "=== Uploading checkpoints to ${OUTPUT_GCS} ==="
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp -r "${OUTPUT_DIR%/}/"* "${OUTPUT_GCS%/}/"
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil -m cp -r "${OUTPUT_DIR%/}/"* "${OUTPUT_GCS%/}/"
  else
    echo "Warning: OUTPUT_GCS is set, but neither gcloud nor gsutil is available." >&2
  fi
fi
