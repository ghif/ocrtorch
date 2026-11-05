"""Character vocabulary and CTC decoding utilities."""

from __future__ import annotations

from collections.abc import Sequence
import torch


def ctc_greedy_decode(
    probabilities: torch.Tensor,
    characters: Sequence[str] | None = None,
    blank_id: int = 0,
) -> list[str] | list[list[int]]:
    """Greedy CTC decode matching PaddleOCR / OCRax blank-first character layout.

    Args:
        probabilities: Tensor of shape [B, T, num_classes] or [B, T].
        characters: Optional list of characters mapped to class index token - 1.
        blank_id: Blank token index, default 0.

    Returns:
        List of decoded text strings (if characters provided) or list of token ID lists.
    """
    if probabilities.ndim == 3:
        token_ids = torch.argmax(probabilities, dim=-1)
    else:
        token_ids = probabilities

    batch_ids = token_ids.detach().cpu().tolist()
    outputs = []
    for row in batch_ids:
        previous = None
        chars = []
        for token in row:
            if token != previous and token != blank_id:
                if characters is not None:
                    if token - 1 >= len(characters):
                        raise ValueError(f"Token ID {token} exceeds dictionary size {len(characters)}")
                    chars.append(characters[token - 1])
                else:
                    chars.append(token)
            previous = token
        outputs.append("".join(chars) if characters is not None else chars)
    return outputs
