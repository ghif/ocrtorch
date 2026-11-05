"""Tests for PP-OCRv6 model output shapes."""

import pytest
import torch

from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


def test_model_forward_eval_shape():
    model = PP_OCRv6MediumRec(num_classes=18710)
    model.eval()
    dummy_input = torch.randn(2, 3, 48, 320)
    with torch.no_grad():
        output = model(dummy_input)
    # Eval mode returns probabilities
    assert output.shape == (2, 40, 18710)
    # Probabilities sum to ~1 along vocabulary axis
    sums = output.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_model_forward_train_shape():
    model = PP_OCRv6MediumRec(num_classes=18710)
    model.train()
    dummy_input = torch.randn(2, 3, 48, 320)
    output = model(dummy_input)
    # Train mode returns raw CTC logits
    assert output.shape == (2, 40, 18710)


def test_model_intermediates_shapes():
    model = PP_OCRv6MediumRec(num_classes=18710)
    model.eval()
    dummy_input = torch.randn(1, 3, 48, 320)
    with torch.no_grad():
        intermediates = model.forward_intermediates(dummy_input)

    assert intermediates["backbone_stem"].shape == (1, 128, 12, 80)
    assert intermediates["backbone_blocks2"].shape == (1, 128, 12, 80)
    assert intermediates["backbone_blocks3"].shape == (1, 256, 12, 80)
    assert intermediates["backbone_blocks4"].shape == (1, 512, 6, 80)
    assert intermediates["backbone_blocks5"].shape == (1, 768, 3, 80)
    assert intermediates["backbone_pool"].shape == (1, 768, 1, 40)
    assert intermediates["neck_reduce"].shape == (1, 192, 1, 40)
    assert intermediates["neck_local"].shape == (1, 192, 1, 40)
    assert intermediates["neck_local_sum"].shape == (1, 192, 1, 40)
    assert intermediates["neck_block_0"].shape == (1, 40, 192)
    assert intermediates["neck_block_1"].shape == (1, 40, 192)
    assert intermediates["neck_norm"].shape == (1, 40, 192)
    assert intermediates["neck_out"].shape == (1, 192, 1, 40)
    assert intermediates["sequence"].shape == (1, 40, 192)
    assert intermediates["ctc_logits"].shape == (1, 40, 18710)
    assert intermediates["probabilities"].shape == (1, 40, 18710)
