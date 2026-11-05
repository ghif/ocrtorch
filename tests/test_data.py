"""Tests for image preprocessing and CTC greedy decoding."""

import numpy as np
import pytest
import torch
from PIL import Image

from ocrtorch.data.images import resize_rec_image, to_torch_tensor
from ocrtorch.data.vocab import ctc_greedy_decode


def test_resize_rec_image_dimensions_and_normalization():
    # Create test RGB image 100x200
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    tensor, resized_w = resize_rec_image(img, image_height=48, image_width=320)

    assert tensor.shape == (3, 48, 320)
    assert 0 < resized_w <= 320
    # Expected height scaled: 48 * 200 / 100 = 96
    assert resized_w == 96
    # Values normalized to [-1, 1]
    assert np.all(tensor >= -1.0) and np.all(tensor <= 1.0)
    # Right padding area should be -1.0 (from 0.0 before norm)
    assert np.allclose(tensor[:, :, resized_w:], -1.0)


def test_to_torch_tensor():
    img = Image.new("RGB", (150, 50), color=(255, 255, 255))
    torch_tensor = to_torch_tensor(img)
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch_tensor.shape == (1, 3, 48, 320)
    assert torch_tensor.dtype == torch.float32


def test_ctc_greedy_decode_token_merging_and_blanks():
    # Batch with 1 sequence: tokens [0, 1, 1, 0, 2, 2, 0, 1]
    # Blank is 0. Expected decoded IDs: [1, 2, 1]
    token_probs = torch.zeros(1, 8, 4)
    tokens = [0, 1, 1, 0, 2, 2, 0, 1]
    for t, tok in enumerate(tokens):
        token_probs[0, t, tok] = 1.0

    decoded_ids = ctc_greedy_decode(token_probs)
    assert decoded_ids == [[1, 2, 1]]

    # With characters dictionary (token 1 -> 'a', token 2 -> 'b')
    characters = ["a", "b", "c"]
    decoded_text = ctc_greedy_decode(token_probs, characters=characters)
    assert decoded_text == ["aba"]
