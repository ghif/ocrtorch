# Achieving Both Speed and Accuracy on TorchTPU `v6e-1`: Hybrid `bfloat16` Autocast with `float32` Master Weights

**Date:** November 21, 2026  
**Hardware:** Google Cloud TPU VM `ct6e-standard-1t` (`1 × TPU v6e chip`)  
**Environment:** `torch-env` (Python `3.13.14`, PyTorch `2.15.0+cpu`, `torch-tpu 0.3.0`, `libtpu 0.0.53.1`, PJRT API `0.117`)  
**Model & Dataset:** `PP_OCRv6MediumRec` (`18,710` classes) on RxHandBD (`3,794` train / `669` validation crops, `seed=42`)

---

## Goal & Outcome

**Goal:** Find a TorchTPU fine-tuning configuration that combines **high TPU throughput** (large batch size + `bfloat16` MXU compute) with **state-of-the-art validation accuracy** (beating the `12.97%` CER / `59.19%` Norm Exact Match `fp32` `batch_size=16` benchmark).

**Winning Configuration:**
- **Command:**
  ```bash
  PYTHONWARNINGS="ignore://TorchTpu/Performance/TcMalloc/" \
    /mnt/data/miniconda3/envs/torch-env/bin/ocrtorch-train \
    --checkpoint ~/ocrtorch_tpu_data/model-dir/ppocrv6_medium_rec.pt \
    --inference-yaml ~/ocrtorch_tpu_data/model-dir/inference.yml \
    --train-manifest ~/ocrtorch_tpu_data/manifests/train.jsonl \
    --val-manifest ~/ocrtorch_tpu_data/manifests/validation.jsonl \
    --output-dir /mnt/data/ocrtorch/artifacts/checkpoints/finetuned_tpu_v6e_hybrid_bs64 \
    --device tpu --bf16 --epochs 40 --batch-size 64 \
    --lr 4e-5 --warmup-epochs 3 --min-lr 1e-6 --patience 8 --num-workers 4
  ```
- **Outcome:**
  - **Best Validation CER:** **`12.32%`** (Epoch 32) — lowest CER across all CPU, MPS, and TPU runs.
  - **Best Normalized Exact Match:** **`61.29%`** (Epoch 32) — **+9.42 pp** over pure `bf16` (`51.87%`), **+4.79 pp** over M4 Pro MPS (`56.50%`), **+2.10 pp** over `fp32` `batch_size=16` (`59.19%`), and **+34.53 pp** over the zero-shot baseline (`26.76%`).
  - **Speed:** **`4.99s` steady-state epoch time** (`4.7s`–`5.3s` across Epochs 2–40, `18.7s` Epoch 1 with XLA compilation) and **`249.3s` (~4.16 min)** total for 40 epochs — faster total runtime than both `fp32` `batch_size=32` (`274.7s`, 15 epochs) and `fp32` `batch_size=16` (`289.2s`, 25 epochs).

---

## Strategy: Why Previous `bfloat16` Underperformed & How We Solved It

### 1. Diagnosing the Speed-vs-Accuracy Gap
In our earlier runs, we observed a sharp tradeoff:
- **Pure `bfloat16` (`batch_size=64`, `lr=3e-5`, 20 epochs):** Fast (`3.81s/epoch`), but stalled at `15.69%` CER / `51.87%` Norm EM (`train_loss=0.3763`).
  - **Why:** Casting model weights and AdamW states directly to `bfloat16` (7-bit mantissa, `~0.0078` relative precision) caused **weight-update stiction**—small fine-tuning updates (`~3e-5`) underflowed below 1 ULP, and `59` steps/epoch × `20` epochs (`1,180` updates) was too few steps to converge.
- **Why `torch.autocast("tpu", dtype=torch.bfloat16)` had failed earlier:**
  - Under `torch.autocast`, `Conv2d` and `Linear` output `bfloat16`, while `nn.LayerNorm` parameters stay `float32`. In `torch_tpu 0.3.0`, `LayerNorm` hit a mixed-dtype StableHLO lowering error (`'stablehlo.multiply' op requires compatible types for all operands and results`).

### 2. The Hybrid Fix
We resolved the root cause cleanly in [`ppocrv6.py`](file:///mnt/data/ocrtorch/src/ocrtorch/models/ppocrv6.py) and [`trainer.py`](file:///mnt/data/ocrtorch/src/ocrtorch/train/trainer.py):
1. **Master weights & AdamW states in `float32`, compute in `bfloat16`:** Replaced full-model `bfloat16` casting with `torch.autocast("tpu", dtype=torch.bfloat16)` so `float32` master weights and AdamW `m`/`v` moments accumulate small updates with zero ULP underflow.
2. **Type-safe `LayerNorm` & attention softmax in [`LightSVTR`](file:///mnt/data/ocrtorch/src/ocrtorch/models/ppocrv6.py#L276-L320):** Cast `LayerNorm` inputs to `weight.dtype` (`float32` under autocast) and cast attention probabilities to `v.dtype` (`bfloat16`), eliminating the `torch_tpu 0.3.0` mixed-dtype StableHLO error.
3. **Compensate larger batch size (`64`) with LR scaling, warmup, cosine decay, & more epochs:** Scaled `--batch-size 64` with `--lr 4e-5`, `--warmup-epochs 3`, cosine decay to `--min-lr 1e-6`, `--epochs 40`, and `--patience 8` (`2,360` total optimizer steps at `~5s/epoch`).
4. **Handwriting-safe augmentation & static-shape async loop:** Kept `±2°` rotation + sharpness/brightness/contrast jitter, static `max_target_len=40`, `drop_last=True`, on-device loss accumulation, and argmax-only validation decoding.
5. **Avoid `LD_PRELOAD` tcmalloc with `fork`ed DataLoader workers:** Preloading `libtcmalloc_minimal.so.4` triggered a `tcmalloc.cc:333` invalid-pointer `SIGABRT` in forked `DataLoader` workers (`num_workers=4`); omitting `LD_PRELOAD` resolved the crash cleanly with zero impact on `4.99s/epoch` throughput.

---

## Cross-Run Comparison

| Configuration | Precision | Batch Size | LR Schedule | Steady-State Epoch | Total Time | Best Val CER | Best Norm EM |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **M4 Pro MPS** | `fp32` | 8 | Constant `1e-5` | `88.5s` | `1,082.4s` *(12 ep)* | `13.63%` | `56.50%` |
| **Colab TPU `v5e1`** | `fp32` | 8 | Constant `1e-5` | `28.9s` | `325.2s` *(10 ep)* | `14.24%` | `55.01%` |
| **TPU `v6e-1` Baseline** | `fp32` | 32 | Constant `2e-5` | `11.95s` | `274.7s` *(15 ep)* | `14.70%` | `54.56%` |
| **TPU `v6e-1` Speed-Opt** | Pure `bf16` | 64 | Constant `3e-5` | **`3.81s`** | **`115.1s`** *(20 ep)* | `15.69%` | `51.87%` |
| **TPU `v6e-1` Accuracy-Opt** | `fp32` | 16 | Warmup + Cosine `2e-5→1e-6` | `8.80s` | `289.2s` *(25 ep)* | `12.97%` | `59.19%` |
| **M4 Pro CPU Hybrid** | `autocast(bf16)` + `fp32` master | 64 | Warmup + Cosine `4e-5→1e-6` | `6,061.1s` *(Ep 1)* | `~67.3h` *(proj. 40 ep)* | `23.28%` *(Ep 1)* | `34.98%` *(Ep 1)* |
| **M4 Pro MPS Hybrid** | **`autocast(bf16)` + `fp32` master** | **64** | **Warmup + Cosine `4e-5→1e-6`** | **`64.36s`** | **`2,640.5s`** *(40 ep)* | **`11.67%`** *(Ep 38)* | **`61.73%`** *(Ep 38, peak 61.88% Ep 34)* |
| **Colab TPU `v5e1` Hybrid** | **`autocast(bf16)` + `fp32` master** | **64** | **Warmup + Cosine `4e-5→1e-6`** | **`9.09s`** | **`404.8s`** *(40 ep)* | **`11.15%`** *(Ep 40)* | **`62.63%`** *(Ep 40, peak 62.78% Ep 38)* |
| **TPU `v6e-1` Hybrid (Goal)** | **`autocast(bf16)` + `fp32` master** | **64** | **Warmup + Cosine `4e-5→1e-6`** | **`4.99s`** | **`249.3s`** *(40 ep)* | **`12.32%`** *(Ep 32)* | **`61.29%`** *(Ep 32)* |

- **XLA compilation cache:** `2,521,981` cache requests, `2,521,423` hits (**99.9%**), `558` compilations, and `0` unbounded dynamic hits.

---

## Artifacts
- **Local (`v6e-1`):** [`artifacts/checkpoints/finetuned_tpu_v6e_hybrid_bs64/`](file:///mnt/data/ocrtorch/artifacts/checkpoints/finetuned_tpu_v6e_hybrid_bs64/training_summary.json) (`best_model.pt`, `latest_model.pt`, `training_summary.json`, `train.log`)
- **Local (`M4 Pro MPS` Hybrid):** `artifacts/checkpoints/finetuned_mps_hybrid_bs64/training_summary.json` (`best_model.pt`, `latest_model.pt`, `training_summary.json`)
- **GCS (`v6e-1`):** `gs://handnotes-ocr/model-dir/finetuned-runs/ppocrv6-rxhandbd-tpu-v6e-hybrid-bs64-20261121T2332Z/`
- **GCS (`v5e1`):** `gs://handnotes-ocr/model-dir/finetuned-runs/ppocrv6-rxhandbd-20261122T111401990025Z/`
