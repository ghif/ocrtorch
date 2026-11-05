# OCRtorch: High-Fidelity PyTorch Port of PP-OCRv6 for TorchTPU

[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg)](https://pytorch.org/)
[![Parity: 100%](https://img.shields.io/badge/Paddle--PyTorch%20Parity-40%2F40%20tokens-brightgreen.svg)](#numerical-parity-verification)
[![TorchTPU Ready](https://img.shields.io/badge/TorchTPU-Ready-blue.svg)](https://pytorch.org/blog/introducing-torchtpu/)

**OCRtorch** is a high-fidelity, modular PyTorch implementation of Baidu's production-grade [PP-OCRv6](https://arxiv.org/abs/2606.13108) text recognition architecture. Developed for the **TorchTPU Sprint**, OCRtorch provides exact mathematical parity against Paddle reference weights, reproducible data preprocessing, and static-sequence compilation design optimized for Google Cloud TPUs via [TorchTPU](https://pytorch.org/blog/introducing-torchtpu/).

---

## Architecture Overview

```mermaid
flowchart LR
    classDef input fill:#f1f5f9,stroke:#64748b,stroke-width:1px,color:#334155;
    classDef backbone fill:#e0f2fe,stroke:#0284c7,stroke-width:1.5px,color:#0369a1;
    classDef neck fill:#ede9fe,stroke:#6366f1,stroke-width:1.5px,color:#3730a3;
    classDef head fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#92400e;
    classDef loss fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d;

    IN["Input Crop<br/>[B, 3, 48, 320]"]:::input --> BB["PPLCNetV4 Backbone<br/>RepDW, SE, ConvBN<br/>Pool → [B, 768, 1, 40]"]:::backbone
    BB --> NK["LightSVTR Neck<br/>1×7 Local Conv + 2× Self-Attention<br/>+ LayerNorm → [B, 192, 1, 40]"]:::neck
    NK --> HD["Linear CTC Head<br/>192 → 18,710 classes<br/>Logits: [B, 40, 18710]"]:::head
    HD --> LS["CTCLossStep<br/>Static Length Alignment"]:::loss
```

1. **Backbone (`PPLCNetV4Rec`)**:
   - Depthwise separable convolutions with residual RepDW connections.
   - Squeeze-and-Excitation (`F.hardsigmoid` channel recalibration).
   - Adaptive/valid pooling collapsing height to 1 and width to $T=40$ sequence steps.
2. **Neck (`LightSVTR`)**:
   - 1×7 horizontal local depthwise convolution for neighboring stroke context.
   - 2× global multi-head self-attention blocks (`heads=8, dim=192`) with `prenorm=False` ordering.
   - Residual skip projection from backbone features.
3. **Head (`CTCHead`)**:
   - Linear classification mapping from 192 features to 18,710 vocabulary classes.

---

## Numerical Parity Verification

OCRtorch was verified against reference activations across all 16 intermediate stages using `tests/test_parity.py`:

| Stage | Output Shape | Max Absolute Diff | Mean Absolute Diff |
| :--- | :---: | :---: | :---: |
| `backbone_stem` | `[1, 128, 12, 80]` | $4.00 \times 10^{-5}$ | $8.89 \times 10^{-7}$ |
| `backbone_blocks2` | `[1, 128, 12, 80]` | $1.06 \times 10^{-5}$ | $3.13 \times 10^{-7}$ |
| `backbone_blocks3` | `[1, 256, 12, 80]` | $9.04 \times 10^{-6}$ | $3.57 \times 10^{-7}$ |
| `backbone_blocks4` | `[1, 512, 6, 80]` | $1.40 \times 10^{-5}$ | $6.97 \times 10^{-7}$ |
| `backbone_blocks5` | `[1, 768, 3, 80]` | $5.26 \times 10^{-5}$ | $2.14 \times 10^{-6}$ |
| `backbone_pool` | `[1, 768, 1, 40]` | $2.71 \times 10^{-5}$ | $1.69 \times 10^{-6}$ |
| `neck_reduce` | `[1, 192, 1, 40]` | $8.58 \times 10^{-6}$ | $7.72 \times 10^{-7}$ |
| `neck_local` | `[1, 192, 1, 40]` | $1.12 \times 10^{-5}$ | $5.12 \times 10^{-7}$ |
| `neck_local_sum` | `[1, 192, 1, 40]` | $1.20 \times 10^{-5}$ | $1.09 \times 10^{-6}$ |
| `neck_block_0` | `[1, 40, 192]` | $1.23 \times 10^{-5}$ | $2.36 \times 10^{-6}$ |
| `neck_block_1` | `[1, 40, 192]` | $1.95 \times 10^{-5}$ | $3.42 \times 10^{-6}$ |
| `neck_norm` | `[1, 40, 192]` | $8.10 \times 10^{-5}$ | $7.78 \times 10^{-6}$ |
| `neck_out` | `[1, 192, 1, 40]` | $8.89 \times 10^{-5}$ | $9.34 \times 10^{-6}$ |
| `sequence` | `[1, 40, 192]` | $8.89 \times 10^{-5}$ | $9.34 \times 10^{-6}$ |
| `ctc_logits` | `[1, 40, 18710]` | $8.42 \times 10^{-5}$ | $6.49 \times 10^{-6}$ |
| `probabilities` | `[1, 40, 18710]` | $2.90 \times 10^{-5}$ | $1.08 \times 10^{-9}$ |
| **Argmax Token Agreement** | **40 / 40 steps** | — | **100.00%** |

---

## Quickstart

### 1. Installation

Activate your conda environment and install `ocrtorch` in editable mode:

```bash
conda activate torch-env
cd /Users/mghifary/Work/Code/AI/ocr/ocrtorch
pip install -e .
```

### 2. Run Parity Tests

```bash
pytest tests/ -v
```

---

## TorchTPU Execution Guide

To train or fine-tune PP-OCRv6 on Google Cloud TPUs (`v4`, `v5e`, `v5p`, or `v6e`) using TorchTPU:

### 1. TPU VM Environment Setup
```bash
pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### 2. Targeting the TPU Device
```python
import torch
import torch_tpu
from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec
from ocrtorch.train.loss import CTCLossStep

# 1. Select the native TPU device
device = torch.device("tpu")

# 2. Instantiate and compile model
model = PP_OCRv6MediumRec().to(device)
compiled_model = torch.compile(model, backend="openxla")

# 3. Setup optimizer and static loss step
optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)
criterion = CTCLossStep().to(device)

# 4. Forward & backward step with static shapes
images = torch.randn(16, 3, 48, 320, device=device)
targets = torch.randint(1, 18700, (16, 20), device=device)
input_lengths = torch.full((16,), 40, dtype=torch.long, device=device)
target_lengths = torch.full((16,), 20, dtype=torch.long, device=device)

logits = compiled_model(images)
loss = criterion(logits, targets, input_lengths, target_lengths)
loss.backward()
optimizer.step()
```

> [!TIP]
> **Static Shape Principle for TPUs:** Always keep batch sizes and image dimensions fixed ($[B, 3, 48, 320]$) and use padded label sequences. This ensures OpenXLA compiles the computational graph once, avoiding compilation stalls.
