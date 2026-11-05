"""OCR text normalization and corpus metrics."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def normalize_text(text: str) -> str:
    """Normalize case and collapse whitespace for tolerant comparisons."""
    return " ".join(text.casefold().split())


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Return Levenshtein edit distance using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Compute corpus CER after case and whitespace normalization."""
    total_distance = 0
    total_reference_chars = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        norm_ref = normalize_text(reference)
        norm_hyp = normalize_text(hypothesis)
        total_distance += edit_distance(norm_ref, norm_hyp)
        total_reference_chars += len(norm_ref)
    if total_reference_chars == 0:
        return 0.0 if total_distance == 0 else math.inf
    return total_distance / total_reference_chars


def exact_match_accuracy(references: Iterable[str], hypotheses: Iterable[str]) -> tuple[float, float]:
    """Compute raw and normalized exact match accuracy.

    Returns:
        (raw_accuracy, normalized_accuracy) as floats in [0.0, 1.0].
    """
    raw_matches = 0
    norm_matches = 0
    count = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        count += 1
        if ref == hyp:
            raw_matches += 1
        if normalize_text(ref) == normalize_text(hyp):
            norm_matches += 1
    if count == 0:
        return 0.0, 0.0
    return raw_matches / count, norm_matches / count
