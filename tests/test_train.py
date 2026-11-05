"""Unit test for CTCLossStep."""

import pytest
import torch

from ocrtorch.train.loss import CTCLossStep


def test_ctc_loss_step_computation_and_backward():
    ctc_step = CTCLossStep(blank_id=0)

    b, t, num_classes = 2, 40, 18710
    logits = torch.randn(b, t, num_classes, requires_grad=True)
    # Targets for 2 samples
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.long)
    input_lengths = torch.tensor([40, 40], dtype=torch.long)
    target_lengths = torch.tensor([4, 2], dtype=torch.long)

    loss = ctc_step(logits, targets, input_lengths, target_lengths)
    assert loss.ndim == 0
    assert not torch.isnan(loss) and not torch.isinf(loss)

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
