"""Unit test for CTCLossStep and EarlyStopping."""

import pytest
import torch

from ocrtorch.train.loss import CTCLossStep
from ocrtorch.train.trainer import EarlyStopping


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


def test_early_stopping_behavior():
    stopper = EarlyStopping(patience=3, min_delta=1e-3)

    # Initial score
    assert stopper.step(0.40) is True
    assert stopper.should_stop is False
    assert stopper.counter == 0

    # Improvement
    assert stopper.step(0.35) is True
    assert stopper.should_stop is False
    assert stopper.counter == 0

    # No improvement: 1st time
    assert stopper.step(0.355) is False
    assert stopper.counter == 1
    assert stopper.should_stop is False

    # No improvement: 2nd time
    assert stopper.step(0.360) is False
    assert stopper.counter == 2
    assert stopper.should_stop is False

    # No improvement: 3rd time -> Early stopping triggered!
    assert stopper.step(0.359) is False
    assert stopper.counter == 3
    assert stopper.should_stop is True
