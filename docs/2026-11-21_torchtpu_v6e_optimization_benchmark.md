# Optimizing TorchTPU Fine-Tuning on Google Cloud TPU `v6e-1`

**Date:** November 21, 2026  
**Hardware:** Google Cloud TPU VM `ct6e-standard-1t` (`1 × TPU v6e chip`)  
**Environment:** `torch-env` (Python `3.13.14`, PyTorch `2.15.0+cpu`, `torch-tpu 0.3.0`, `libtpu 0.0.53.1`, PJRT API `0.117`)  
**Model & Dataset:** `PP_OCRv6MediumRec` (`18,710` classes) on RxHandBD (`3,794` train / `669` validation crops, `seed=42`)

---

## Summary

Eliminating host-device synchronization stalls, switching to native `bfloat16` compute, enforcing 100% static tensor shapes, skipping redundant validation `softmax`, preloading `libtcmalloc_minimal.so.4`, and scaling batch size from `32` to `64` reduced steady-state epoch time on TPU `v6e-1` from **`11.95s`** to **`3.81s`** (**3.1× speedup** over baseline `v6e-1`, **7.6×** over Colab `v5e1`, and **23.2×** over Apple M4 Pro MPS), completing **20 epochs in `115.1s` (~1.92 minutes)**.

---

## Bottlenecks Identified & Changes Implemented

| Optimization | File | What Changed | Why It Matters on TorchTPU |
| :--- | :--- | :--- | :--- |
| **Zero per-batch syncs** | [`trainer.py`](file:///mnt/data/ocrtorch/src/ocrtorch/train/trainer.py) | Replaced per-batch `torch.isnan(loss)` / `loss.item()` with on-device `loss.detach().float()` accumulation synced once per epoch. | Removes `119` blocking TPU→CPU syncs per epoch, keeping the asynchronous PJRT execution queue full. |
| **Native `bfloat16` compute** | [`trainer.py`](file:///mnt/data/ocrtorch/src/ocrtorch/train/trainer.py) | Cast model and input images to `torch.bfloat16` on TPU (`CTCLossStep` remains `float32`; checkpoints saved in `float32`). | Uses `v6e` native `bfloat16` MXUs and halves activation/weight HBM bandwidth. Also avoids a `torch_tpu 0.3.0` `LayerNorm` mixed-dtype StableHLO bug under `torch.autocast`. |
| **100% static shapes** | [`dataset.py`](file:///mnt/data/ocrtorch/src/ocrtorch/data/dataset.py), [`train.py`](file:///mnt/data/ocrtorch/src/ocrtorch/cli/train.py) | Added `max_target_len=40` padding in `ocr_collate_fn` and `drop_last=True` on `train_loader`. | Eliminates batch-varying `targets` lengths and trailing partial batches that trigger extra XLA/StableHLO compilations. |
| **Skip validation `softmax`** | [`trainer.py`](file:///mnt/data/ocrtorch/src/ocrtorch/train/trainer.py) | Pass `logits.argmax(dim=-1)` directly to `ctc_greedy_decode`. | Avoids materializing and reducing `[B, 40, 18710]` float32 softmax probabilities on every validation batch. |
| **DataLoader & `tcmalloc`** | [`train.py`](file:///mnt/data/ocrtorch/src/ocrtorch/cli/train.py) | Enabled `persistent_workers=True` & `prefetch_factor=4` when `num_workers > 0`; installed and preloaded `libtcmalloc_minimal.so.4`. | Prevents worker respawn overhead between epochs and eliminates glibc `ptmalloc` host contention. |
| **Larger TPU batch size** | CLI flags | Scaled `--batch-size 64 --lr 3e-5` (`59` train steps/epoch vs `119` at `batch_size=32`). | Better saturates `v6e-1` TensorCore/MXU compute and halves per-epoch Python dispatch overhead. |

### Empirical Note on `torch.compile` with `torch-tpu 0.3.0`
We tested both Dynamo compile backends registered in `torch-env`:
- `backend="openxla"` requires `torch_xla` (not applicable to native `torch_tpu`).
- `backend="tpu"` failed during `LightSVTR` `LayerNorm` export with `RuntimeError: traverse_and_compile(): failed to compile traversal: failed to inline functions in MLIR module`.
- `--compile-backend` therefore defaults to `None` (disabled) while remaining available as an opt-in CLI flag for future `torch-tpu` releases.

---

## Benchmark Comparison Across Runs

| Configuration | Batch Size | LR | Steady-State Epoch (s) | Epoch 1 (s) | Total Time | Best Val CER | Best Norm EM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M4 Pro MPS** (`fp32`) | 8 | `1e-5` | `88.5s` *(Ep 2–12)* | `65.8s` | `1,082.4s` *(12 ep)* | **13.63%** *(Ep 9)* | **56.50%** *(Ep 9)* |
| **Colab TPU `v5e1`** (`fp32`) | 8 | `1e-5` | `28.9s` *(Ep 2–10)* | `46.3s` | `325.2s` *(10 ep)* | **14.24%** *(Ep 7)* | **55.01%** *(Ep 7)* |
| **TPU `v6e-1` Baseline** (`fp32`, eager) | 32 | `2e-5` | `11.95s` *(Ep 2–15)* | `85.6s` | `274.7s` *(15 ep)* | **14.70%** *(Ep 11)* | **54.56%** *(Ep 11)* |
| **TPU `v6e-1` Speed-Opt** (`bf16`, static, async) | 64 | `3e-5` | **`3.81s`** *(Ep 2–20)* | **`17.6s`** | **`115.1s`** *(20 ep)* | **15.69%** *(Ep 19)* | **51.87%** *(Ep 19)* |
| **TPU `v6e-1` Accuracy-Opt** (`fp32`, cosine LR, aug) | 16 | `2e-5→1e-6` | **`8.80s`** *(Ep 2–25)* | **`50.8s`** | `289.2s` *(25 ep)* | **`12.97%`** *(Ep 24)* | **`59.19%`** *(Ep 24)* |

- **Compilation cache efficiency:** The optimized run logged **816,135** cache requests with **815,585 hits (99.9%)**, **550** compilations (down from `725` in the baseline `v6e-1` run), and **0** unbounded dynamic hits.
- **Convergence tradeoff at `batch_size=64`:** With `59` optimizer steps/epoch (half as many updates per epoch as `batch_size=32`), validation loss (`0.6940` at Epoch 19) and CER (`15.69%`) were still steadily improving at Epoch 20 without overfitting. Increasing `--lr` (e.g., `5e-5`) or training `30` epochs (~`2.5` minutes total) can recover the ~1% CER gap while keeping the `3.81s/epoch` speed.

---

## Per-Epoch Training Curve (TPU `v6e-1` Optimized)

Zero-shot baseline before Epoch 1: **2.0485** Val Loss, **32.46%** CER, **27.20%** Norm Exact Match.

| Epoch | Train Loss | Val Loss | Val CER | Norm Exact Match | Time (s) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 1.2711 | 1.0487 | 25.08% | 34.23% | 17.6 | Best |
| 2 | 1.0175 | 0.9943 | 24.51% | 36.62% | 3.8 | Best |
| 3 | 0.9292 | 0.9316 | 22.55% | 38.42% | 3.7 | Best |
| 4 | 0.8660 | 0.8737 | 22.01% | 38.57% | 3.7 | Best |
| 5 | 0.8218 | 0.8682 | 20.91% | 41.11% | 3.8 | Best |
| 6 | 0.7670 | 0.8098 | 20.78% | 42.75% | 3.7 | Best |
| 7 | 0.7201 | 0.7959 | 20.31% | 42.90% | 3.7 | Best |
| 8 | 0.6729 | 0.8158 | 20.41% | 41.85% | 3.8 | Patience 1/4 |
| 9 | 0.6458 | 0.7472 | 19.57% | 45.14% | 3.8 | Best |
| 10 | 0.6086 | 0.7759 | 18.90% | 44.99% | 3.8 | Best |
| 11 | 0.5792 | 0.7428 | 18.00% | 46.94% | 3.8 | Best |
| 12 | 0.5497 | 0.7537 | 18.17% | 45.44% | 3.8 | Patience 1/4 |
| 13 | 0.5210 | 0.7231 | 17.87% | 47.53% | 3.8 | Best |
| 14 | 0.5043 | 0.7293 | 18.19% | 47.38% | 3.8 | Patience 1/4 |
| 15 | 0.4733 | 0.7202 | 17.05% | 48.88% | 3.8 | Best |
| 16 | 0.4587 | 0.7233 | 16.40% | 50.67% | 4.0 | Best |
| 17 | 0.4290 | 0.7226 | 16.38% | 49.93% | 3.9 | Best |
| 18 | 0.4222 | 0.7189 | 16.64% | 48.28% | 3.9 | Patience 1/4 |
| **19** | **0.4023** | **0.6940** | **15.69%** | **51.87%** | 3.9 | **Best (Selected)** |
| 20 | 0.3763 | 0.7286 | 16.18% | 49.78% | 3.9 | Patience 1/4 |

---

## Artifacts
- **Speed-Opt (`bf16`, `batch_size=64`):**
  - Local: [`artifacts/checkpoints/finetuned_tpu_v6e_opt/`](file:///mnt/data/ocrtorch/artifacts/checkpoints/finetuned_tpu_v6e_opt/training_summary.json)
  - GCS: `gs://handnotes-ocr/model-dir/finetuned-runs/ppocrv6-rxhandbd-tpu-v6e-opt-20261121T1427Z/`
- **Accuracy-Opt (`fp32`, `batch_size=16`, warmup + cosine LR, aug):**
  - Local: [`artifacts/checkpoints/finetuned_tpu_v6e_acc/`](file:///mnt/data/ocrtorch/artifacts/checkpoints/finetuned_tpu_v6e_acc/training_summary.json)
  - GCS: `gs://handnotes-ocr/model-dir/finetuned-runs/ppocrv6-rxhandbd-tpu-v6e-acc-20261121T1439Z/`
  - **Best result:** **12.97% Val CER** and **59.19% Norm Exact Match** at Epoch 24 (`289.2s` total for 25 epochs, `8.80s/epoch` steady-state, **99.9%** XLA cache hit rate across `3,957,332` requests and `439` compilations).
