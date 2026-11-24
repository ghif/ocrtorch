# Fine-Tuning PP-OCRv6 on Buku KIA Page 3 with TorchTPU (`v6e-1`)

**Date:** November 24, 2026  
**Hardware:** Google Cloud TPU VM `ct6e-standard-1t` (`1 × TPU v6e chip`)  
**Environment:** `torch-env` (Python `3.13.14`, PyTorch `2.15.0+cpu`, `torch-tpu 0.3.0`, `libtpu 0.0.53.1`, PJRT API `0.117`)  
**Model:** `PP_OCRv6MediumRec` (`18,710` classes, `34.5M` params)  
**Dataset:** `bukukia_page3` (Handwritten Indonesian maternal/child health book — `blood_pressure`, `dates`, `hemoglobin`)

---

## Executive Summary

Fine-tuning `PP_OCRv6MediumRec` on **Buku KIA Page 3 (`bukukia_page3`)** using TorchTPU's hybrid `bfloat16` autocast (`float32` master weights + `bfloat16` MXU compute) with anti-forgetting regularization (`L2-SP`, differential backbone LR, unseen CTC token protection) and field-balanced sampling achieved strong generalization in **593.4 seconds (~9.89 minutes)** across 30 epochs:

- **Held-Out Test (`2,215` crops):** Reduced Character Error Rate (CER) from **`24.93%`** (zero-shot) to **`4.00%`**, and lifted Normalized Exact Match from **`28.53%`** to **`80.59%`** (**+52.06 pp**).
- **Validation (`1,313` crops):** Reduced CER from **`26.07%`** (zero-shot) to **`4.32%`** (Epoch 28), and lifted Normalized Exact Match from **`27.57%`** to **`79.89%`** (peaking at **`80.05%`** at Epoch 24).
- **Zero overfitting:** Held-out test accuracy (`4.00%` CER / `80.59%` Norm EM) slightly outperformed validation (`4.32%` CER / `79.89%` Norm EM), with `val_loss` remaining flat around `0.173–0.178` from Epoch 12 through Epoch 30.

---

## Dataset & Task-Grouped Split

From [`data/manifests/bukukia_page3/split_metadata.json`](file:///mnt/data/ocrtorch/data/manifests/bukukia_page3/split_metadata.json), crops are split by `task_id` (`seed=140`, `val_fraction=0.15`) so no document/task leaks across splits:

| Split | Unique Tasks | Total Crops | `dates` | `blood_pressure` | `hemoglobin` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 863 | **7,389** | 4,381 *(59.3%)* | 2,681 *(36.3%)* | 327 *(4.4%)* |
| **Validation** | 152 | **1,313** | 787 *(59.9%)* | 470 *(35.8%)* | 56 *(4.3%)* |
| **Test (Held-out)** | 254 | **2,215** | 1,323 *(59.7%)* | 787 *(35.5%)* | 105 *(4.7%)* |

Because `hemoglobin` represents only `4.4%` of training crops, the training loader uses **field-balanced sampling (`balance_alpha=0.5`)** to upweight minority fields without starving `dates` and `blood_pressure`.

---

## Configuration & Anti-Forgetting Strategy

Training was launched with [`configs/bukukia_page3_tpu_v6e.yaml`](file:///mnt/data/ocrtorch/configs/bukukia_page3_tpu_v6e.yaml):

| Parameter | Value | Role |
| :--- | :---: | :--- |
| `device` / `bf16` | `tpu` / `true` | Hybrid `torch.autocast("tpu", dtype=torch.bfloat16)` + `float32` master weights/AdamW |
| `batch_size` / `drop_last` | `64` / `true` | `115` static-shape steps/epoch; `max_target_len=40` |
| `epochs` / `patience` | `30` / `6` | Cosine annealing (`warmup_epochs=3`, `min_lr=1e-6`) |
| `lr` / `backbone_lr_scale` | `2.0e-5` / `0.1` | Neck & CTC head peak at `2.0e-5`; `PPLCNetV4` backbone peaks at `2.0e-6` |
| `l2_sp_weight` | `5.0e-4` | L2-SP anchor penalty toward pretrained weights |
| `protect_unseen_ctc` | `true` | Freezes gradients on unseen CTC rows (`29 / 18,710` active token IDs) |
| `balance_fields` / `balance_alpha` | `true` / `0.5` | Inverse-frequency square-root field sampling |

---

## Held-Out Test & Validation Evaluation (Zero-Shot vs. Fine-Tuned)

Evaluated on TPU `v6e-1` comparing pretrained [`ppocrv6_medium_rec.pt`](file:///mnt/data/ocrtorch/artifacts/ppocrv6_medium_rec.pt) against fine-tuned [`best_model.pt`](file:///mnt/data/ocrtorch/artifacts/checkpoints/bukukia_page3_tpu_v6e/best_model.pt) (Epoch 28):

| Split / Field | Crops | Zero-Shot CER | Fine-Tuned CER | Zero-Shot Norm EM | Fine-Tuned Norm EM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation (Overall)** | **1,313** | 26.07% | **4.32%** *(-21.75 pp)* | 27.57% | **79.89%** *(+52.32 pp)* |
| — `blood_pressure` | 470 | 21.82% | **2.72%** | 38.72% | **86.81%** |
| — `dates` | 787 | 28.10% | **5.08%** | 21.35% | **75.35%** |
| — `hemoglobin` | 56 | 27.48% | **4.95%** | 21.43% | **85.71%** |
| **Test (Overall)** | **2,215** | 24.93% | **4.00%** *(-20.93 pp)* | 28.53% | **80.59%** *(+52.06 pp)* |
| — `blood_pressure` | 787 | 24.81% | **3.10%** | 34.43% | **86.91%** |
| — `dates` | 1,323 | 24.88% | **4.34%** | 25.25% | **76.64%** |
| — `hemoglobin` | 105 | 27.36% | **6.13%** | 25.71% | **82.86%** |

---

## Full 30-Epoch Training Trajectory

Zero-shot baseline before Epoch 1: **1.2801** Val Loss, **26.07%** CER, **27.57%** Norm Exact Match.  
*(Note: `Backbone LR` reflects `param_groups[0]` with `backbone_lr_scale=0.1`; neck & CTC head LR is `10×` higher, peaking at `2.00e-5` at Epoch 3.)*

| Epoch | Backbone LR | Train Loss | Val Loss | Val CER | Norm Exact Match | Time (s) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `6.67e-07` | 0.6617 | 0.4621 | 12.91% | 52.09% | 36.4 | Best *(incl. XLA compile)* |
| 2 | `1.33e-06` | 0.3782 | 0.3226 | 9.19% | 62.68% | 18.0 | Best |
| 3 | `2.00e-06` | 0.2734 | 0.2546 | 7.16% | 68.39% | 16.9 | Best |
| 4 | `2.00e-06` | 0.2171 | 0.2317 | 6.45% | 71.52% | 17.9 | Best |
| 5 | `1.99e-06` | 0.1836 | 0.2095 | 6.14% | 72.28% | 19.1 | Best |
| 6 | `1.97e-06` | 0.1629 | 0.2034 | 5.72% | 73.95% | 18.4 | Best |
| 7 | `1.94e-06` | 0.1522 | 0.1916 | 5.35% | 75.63% | 18.7 | Best |
| 8 | `1.90e-06` | 0.1389 | 0.1878 | 5.12% | 76.62% | 18.2 | Best |
| 9 | `1.84e-06` | 0.1300 | 0.1836 | 5.11% | 76.62% | 18.2 | Best |
| 10 | `1.78e-06` | 0.1212 | 0.1822 | 5.15% | 76.54% | 17.8 | Patience 1/6 |
| 11 | `1.70e-06` | 0.1057 | 0.1814 | 4.86% | 77.84% | 17.0 | Best |
| 12 | `1.62e-06` | 0.1061 | 0.1757 | 4.73% | 78.07% | 18.0 | Best |
| 13 | `1.53e-06` | 0.0981 | 0.1759 | 4.67% | 78.67% | 17.9 | Best |
| 14 | `1.43e-06` | 0.0951 | 0.1791 | 4.74% | 77.99% | 18.5 | Patience 1/6 |
| 15 | `1.32e-06` | 0.0938 | 0.1758 | 4.69% | 78.45% | 18.4 | Patience 2/6 |
| 16 | `1.21e-06` | 0.0915 | **0.1734** | 4.68% | 78.45% | 17.8 | Patience 3/6 |
| 17 | `1.11e-06` | 0.0876 | 0.1767 | 4.62% | 78.67% | 18.6 | Best |
| 18 | `9.95e-07` | 0.0810 | 0.1743 | 4.54% | 79.13% | 18.8 | Best |
| 19 | `8.85e-07` | 0.0726 | 0.1736 | 4.54% | 78.90% | 18.6 | Patience 1/6 |
| 20 | `7.78e-07` | 0.0727 | 0.1738 | 4.42% | 79.82% | 17.9 | Best |
| 21 | `6.74e-07` | 0.0738 | 0.1763 | 4.58% | 78.67% | 18.2 | Patience 1/6 |
| 22 | `5.75e-07` | 0.0700 | 0.1772 | 4.48% | 79.21% | 17.9 | Patience 2/6 |
| 23 | `4.83e-07` | 0.0725 | 0.1745 | 4.39% | 79.66% | 18.4 | Best |
| 24 | `3.98e-07` | 0.0692 | 0.1753 | 4.37% | **80.05%** | 17.3 | Best |
| 25 | `3.22e-07` | 0.0695 | 0.1767 | 4.42% | 79.59% | 16.5 | Patience 1/6 |
| 26 | `2.56e-07` | 0.0642 | 0.1779 | 4.37% | 79.97% | 18.3 | Patience 2/6 |
| 27 | `2.01e-07` | 0.0676 | 0.1749 | 4.38% | 79.51% | 17.1 | Patience 3/6 |
| **28** | `1.57e-07` | 0.0697 | 0.1737 | **4.32%** | 79.89% | 18.3 | **Best (Selected)** |
| 29 | `1.26e-07` | 0.0618 | 0.1753 | 4.36% | 79.66% | 16.6 | Patience 1/6 |
| 30 | `1.06e-07` | 0.0645 | 0.1754 | 4.33% | 79.97% | 17.2 | Patience 2/6 |

---

## TPU `v6e-1` Runtime & XLA Compilation Analysis

- **Epoch timing:** Epoch 1 took **`36.4s`** (including initial XLA/StableHLO graph compilation), after which Epochs 2–30 averaged **`17.91s` per epoch** (`16.5s`–`19.1s`) across `115` training batches + `21` validation batches per epoch.
- **XLA compilation cache:** Logged **`8,224,641`** cache requests with **`8,223,648` hits (99.9%)**, **`993`** compilations, and **`0` unbounded dynamic hits**.

---

## Artifacts & Reproducibility

- **Command:**
  ```bash
  PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/" ocrtorch-train \
    --config configs/bukukia_page3_tpu_v6e.yaml \
    --checkpoint artifacts/ppocrv6_medium_rec.pt \
    --inference-yaml "${HOME}/.paddlex/official_models/PP-OCRv6_medium_rec/inference.yml" \
    --train-manifest data/manifests/bukukia_page3/train.jsonl \
    --val-manifest data/manifests/bukukia_page3/val.jsonl \
    --output-dir artifacts/checkpoints/bukukia_page3_tpu_v6e
  ```
- **Local checkpoints & summary:** [`artifacts/checkpoints/bukukia_page3_tpu_v6e/`](file:///mnt/data/ocrtorch/artifacts/checkpoints/bukukia_page3_tpu_v6e/training_summary.json) (`best_model.pt`, `latest_model.pt`, `training_summary.json`, `train.log`)
- **GCS backup:** `gs://handnotes-ocr/model-dir/finetuned-runs/ppocrv6-bukukia-page3-tpu-v6e-20261124T1412Z/` (`4` objects, `147.16 MiB`)
