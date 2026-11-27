"""Tests for PyTorch PP_OCRv6MediumDet shape and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch

from ocrtorch.models.ppocrv6_det import PP_OCRv6MediumDet


def test_det_forward_shape_and_range():
    model = PP_OCRv6MediumDet()
    model.eval()
    x = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_det_checkpoint_loads_if_present():
    ckpt = Path(__file__).resolve().parent.parent / "artifacts" / "ppocrv6_medium_det.pt"
    if not ckpt.is_file():
        return
    model = PP_OCRv6MediumDet()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64, dtype=torch.float32))
    assert out.shape == (1, 1, 64, 64)
