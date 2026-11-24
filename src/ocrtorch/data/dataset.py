"""Dataset and manifest loading for PP-OCRv6 fine-tuning."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Sequence
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import yaml

from ocrtorch.data.images import resize_rec_image


def load_character_table(inference_yaml_path: Path | str) -> tuple[list[str], dict[str, int]]:
    """Load character vocabulary from PaddleOCR inference.yml."""
    with open(inference_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    chars = list(config["PostProcess"]["character_dict"])
    # Add space token
    chars.append(" ")
    # Blank ID is 0, dictionary tokens start at 1
    char_to_id = {char: idx + 1 for idx, char in enumerate(chars)}
    return chars, char_to_id


def load_manifest(manifest_path: Path | str) -> list[dict[str, str]]:
    """Load lines from a JSONL manifest file."""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class OCRDataset(Dataset):
    """OCR Dataset for line crops with optional training augmentation."""

    def __init__(
        self,
        manifest_path: Path | str,
        char_to_id: dict[str, int],
        augment: bool = False,
        seed: int = 42,
    ):
        self.rows = load_manifest(manifest_path)
        self.char_to_id = char_to_id
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = row["image"]
        text = row["text"]

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            if self.augment:
                angle = self.rng.uniform(-2.0, 2.0)
                if abs(angle) > 0.1:
                    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
                image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(0.9, 1.1))
                image = ImageEnhance.Contrast(image).enhance(self.rng.uniform(0.9, 1.1))
                image = ImageEnhance.Sharpness(image).enhance(self.rng.uniform(0.8, 1.2))

            tensor_chw, valid_w = resize_rec_image(image)

        # Encode text to IDs
        token_ids = []
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Fallback: ignore or use unknown
                continue

        return {
            "image": torch.from_numpy(tensor_chw).to(torch.float32),
            "token_ids": token_ids,
            "text": text,
        }


def ocr_collate_fn(batch: list[dict], max_target_len: int | None = None) -> dict:
    """Collate batch into padded tensors ready for CTCLoss and TorchTPU."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = [item["text"] for item in batch]
    token_lists = [item["token_ids"] for item in batch]

    target_lengths = torch.tensor([len(tokens) for tokens in token_lists], dtype=torch.long)
    batch_max = max(1, target_lengths.max().item())
    max_len = max(batch_max, max_target_len) if max_target_len is not None else batch_max

    targets = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        if len(tokens) > 0:
            targets[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    # Static input sequence length for PP-OCRv6 medium is 40
    input_lengths = torch.full((len(batch),), 40, dtype=torch.long)

    return {
        "images": images,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "texts": texts,
    }


def build_field_balanced_sampler(
    rows: Sequence[dict[str, str]],
    alpha: float = 0.5,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that upweights minority fields by N_field^(-alpha)."""
    if not rows:
        raise ValueError("Cannot build sampler from empty rows")
    fields = [row.get("field", "__default__") for row in rows]
    counts = Counter(fields)
    weights = [float(counts[f]) ** (-alpha) for f in fields]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )


def collect_active_token_ids(
    rows: Sequence[dict[str, str]],
    char_to_id: dict[str, int],
    include_blank: bool = True,
) -> set[int]:
    """Collect vocabulary token IDs present in manifest texts (plus blank=0 by default)."""
    active: set[int] = {0} if include_blank else set()
    for row in rows:
        for ch in row.get("text", ""):
            if ch in char_to_id:
                active.add(char_to_id[ch])
    return active
