"""Tests for OCRDataset and collation."""

import json
from pathlib import Path
import pytest
import torch
from PIL import Image

from ocrtorch.data.dataset import OCRDataset, ocr_collate_fn


def test_ocr_dataset_and_collate(tmp_path):
    # Create fake image and manifest
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 48), color=(255, 0, 0))
    img.save(img_path)

    manifest_path = tmp_path / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"image": str(img_path), "text": "ab"}) + "\n")

    char_to_id = {"a": 1, "b": 2, " ": 3}
    dataset = OCRDataset(manifest_path, char_to_id, augment=False)
    assert len(dataset) == 1

    item = dataset[0]
    assert item["image"].shape == (3, 48, 320)
    assert item["token_ids"] == [1, 2]
    assert item["text"] == "ab"

    batch = ocr_collate_fn([item, item])
    assert batch["images"].shape == (2, 3, 48, 320)
    assert batch["targets"].shape == (2, 2)
    assert torch.equal(batch["input_lengths"], torch.tensor([40, 40]))
    assert torch.equal(batch["target_lengths"], torch.tensor([2, 2]))
