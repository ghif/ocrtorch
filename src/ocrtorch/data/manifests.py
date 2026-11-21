"""Manifest loading and validation for fine-tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def manifest_hash(path: Path) -> str:
    """Compute sha256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load JSONL manifest."""
    with path.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if not examples:
        raise ValueError(f"Manifest is empty: {path}")
    return examples


def validate_colab_manifests(manifest_dir: Path) -> None:
    """Validate the train/validation split contract before a Colab run."""
    metadata_path = manifest_dir / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Split metadata must be a JSON object: {metadata_path}")
    if metadata.get("dataset") != "RxHandBD" or metadata.get("source") != "Train_Label.csv only":
        raise ValueError("Colab training manifests must be generated from RxHandBD Train_Label.csv only")
    if metadata.get("test_labels_read") is not False:
        raise ValueError("Split metadata must confirm that test labels were not read")

    split_paths = {}
    split_basenames = {}
    for split in ("train", "validation"):
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {path}")
        expected_hash = metadata.get(f"{split}_sha256")
        if not isinstance(expected_hash, str) or manifest_hash(path) != expected_hash:
            raise ValueError(f"{split} manifest does not match {metadata_path}")
        rows = load_manifest(path)
        expected_count = metadata.get(f"{split}_count")
        if type(expected_count) is not int or len(rows) != expected_count:
            raise ValueError(f"{split} manifest row count does not match {metadata_path}")
        images = set()
        basenames = set()
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{split} manifest row {index} must be a JSON object")
            image = row.get("image")
            text = row.get("text")
            if not isinstance(image, str) or not image or not isinstance(text, str) or not text:
                raise ValueError(f"{split} manifest row {index} needs non-empty image and text fields")
            image_path = Path(image)
            if not image_path.name or image_path.parent.name != "Train_Set":
                raise ValueError(f"{split} manifest row {index} must reference an image in Train_Set")
            if image in images or image_path.name in basenames:
                raise ValueError(f"{split} manifest contains duplicate image {image!r}")
            images.add(image)
            basenames.add(image_path.name)
        split_paths[split] = images
        split_basenames[split] = basenames

    if split_paths["train"] & split_paths["validation"]:
        raise ValueError("Train and validation manifests contain overlapping image paths")
    if split_basenames["train"] & split_basenames["validation"]:
        raise ValueError("Train and validation manifests contain overlapping image basenames")
