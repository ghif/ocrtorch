"""CTC loss and training utilities for PyTorch, Apple Silicon MPS, and TorchTPU."""

from __future__ import annotations

import torch
import torch.nn as nn


class CTCLossStep(nn.Module):
    """CTC loss step designed for static shapes, Apple Silicon MPS, and TorchTPU / XLA compilation."""

    def __init__(self, blank_id: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank_id, zero_infinity=zero_infinity, reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            logits: Predicted logits of shape [B, T, num_classes].
            targets: Encoded label targets of shape [B, max_target_len] (or concatenated [sum(target_lengths)]).
            input_lengths: Sequence lengths of predictions [B]. For static PP-OCRv6, all are T=40.
            target_lengths: Sequence lengths of targets [B].

        Returns:
            Scalar mean CTC loss tensor.
        """
        # PyTorch CTCLoss expects log_probs of shape [T, B, num_classes]
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        # Apple Silicon MPS does not yet natively lower aten::_ctc_loss.
        # Fallback loss execution to CPU while preserving gradient backprop on MPS.
        if logits.device.type == "mps":
            loss = self.criterion(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu(),
            )
            return loss.to(logits.device)

        return self.criterion(
            log_probs,
            targets.to(logits.device),
            input_lengths.to(logits.device),
            target_lengths.to(logits.device),
        )
