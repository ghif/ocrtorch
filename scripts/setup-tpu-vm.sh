#!/usr/bin/env bash
set -euo pipefail

# One-time setup & GCS staging for PP-OCRv6 fine-tuning on a Google Cloud TPU VM (e.g. v6e)
# using TorchTPU (https://github.com/google-pytorch/torch_tpu).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
TPU_DATA_DIR="${TPU_DATA_DIR:-${HOME}/ocrtorch_tpu_data}"
MODEL_GCS="${MODEL_GCS:-gs://handnotes-ocr/model-dir}"
CHECKPOINT_GCS="${CHECKPOINT_GCS:-${MODEL_GCS%/}/ppocrv6_medium_rec.pt}"
DATASET_GCS="${DATASET_GCS:-gs://all-datasets/RxHandBD}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

pick_python() {
  for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Error: Python 3.11+ is required on the TPU VM." >&2
  return 1
}

echo "=== 1/4 Setting up Python virtual environment at ${VENV_DIR} ==="
PYTHON_BIN="$(pick_python)"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "=== 2/4 Installing TorchTPU (CPU PyTorch index to avoid CUDA wheels) and ocrtorch ==="
python -m pip install --upgrade \
  --extra-index-url "${TORCH_CPU_INDEX}" \
  "torch>=2.0.0" \
  "torchvision>=0.15.0" \
  "torch-tpu"
python -m pip install -e "${REPO_ROOT}[cloud]"

echo "=== 3/4 Verifying TorchTPU device ==="
python - <<'PY'
import torch
import torch_tpu  # noqa: F401

device = torch.device("tpu")
x = torch.ones((2, 2), device=device)
y = (x + x).cpu()
print(f"Verified TorchTPU device={device}, sample sum={y.sum().item():.1f}")
PY

echo "=== 4/4 Staging model, dataset, and rewritten manifests into ${TPU_DATA_DIR} ==="
MODEL_DIR="${TPU_DATA_DIR}/model-dir"
IMAGE_DIR="${TPU_DATA_DIR}/RxHandBD/Train_Set"
MANIFEST_DIR="${TPU_DATA_DIR}/manifests"
mkdir -p "${MODEL_DIR}" "${IMAGE_DIR}" "${MANIFEST_DIR}"

if command -v gcloud >/dev/null 2>&1; then
  if [[ ! -f "${MODEL_DIR}/inference.yml" ]]; then
    gcloud storage cp "${MODEL_GCS%/}/inference.yml" "${MODEL_DIR}/inference.yml"
  fi
  if [[ ! -f "${MODEL_DIR}/ppocrv6_medium_rec.pt" ]]; then
    gcloud storage cp "${CHECKPOINT_GCS}" "${MODEL_DIR}/ppocrv6_medium_rec.pt"
  fi
  gcloud storage rsync -r "${DATASET_GCS%/}/Train_Set" "${IMAGE_DIR}"
elif command -v gsutil >/dev/null 2>&1; then
  if [[ ! -f "${MODEL_DIR}/inference.yml" ]]; then
    gsutil cp "${MODEL_GCS%/}/inference.yml" "${MODEL_DIR}/inference.yml"
  fi
  if [[ ! -f "${MODEL_DIR}/ppocrv6_medium_rec.pt" ]]; then
    gsutil cp "${CHECKPOINT_GCS}" "${MODEL_DIR}/ppocrv6_medium_rec.pt"
  fi
  gsutil -m rsync -r "${DATASET_GCS%/}/Train_Set" "${IMAGE_DIR}"
else
  echo "Error: Neither 'gcloud' nor 'gsutil' found. Install Google Cloud CLI to stage GCS assets." >&2
  exit 1
fi

export REPO_ROOT TPU_DATA_DIR
python - <<'PY'
import json
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
data_dir = Path(os.environ["TPU_DATA_DIR"])
image_dir = data_dir / "RxHandBD" / "Train_Set"
manifest_out = data_dir / "manifests"

for split in ("train", "validation"):
    src = repo_root / "data" / "manifests" / f"{split}.jsonl"
    dst = manifest_out / f"{split}.jsonl"
    rows = []
    missing = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        local_img = image_dir / Path(record["image"]).name
        if not local_img.is_file():
            missing += 1
        record["image"] = str(local_img)
        rows.append(json.dumps(record, ensure_ascii=False))
    if missing:
        raise FileNotFoundError(f"{missing} images referenced in {src} are missing under {image_dir}")
    dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} records to {dst}")
PY

echo "=== TPU VM setup complete! Run ./scripts/finetune-tpu-vm.sh to start training ==="
