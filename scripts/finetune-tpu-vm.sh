#!/usr/bin/env bash
set -euo pipefail

# Run PP-OCRv6 fine-tuning on a provisioned Google Cloud TPU VM (e.g. v6e) using TorchTPU.
# Prerequisite: run ./scripts/setup-tpu-vm.sh once on the VM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
TPU_DATA_DIR="${TPU_DATA_DIR:-${HOME}/ocrtorch_tpu_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/checkpoints/finetuned_tpu_v6e}"
OUTPUT_GCS="${OUTPUT_GCS:-}"

if [[ -z "${VIRTUAL_ENV:-}" && -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

# TorchTPU recommends preloading libtcmalloc to avoid glibc ptmalloc contention on the host.
TCMALLOC_LIB="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
if [[ -z "${LD_PRELOAD:-}" && -f "${TCMALLOC_LIB}" ]]; then
  export LD_PRELOAD="${TCMALLOC_LIB}"
fi

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
  --checkpoint "${CHECKPOINT_PATH}" \
  --inference-yaml "${INFERENCE_YAML}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --device tpu \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --patience 3 \
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
