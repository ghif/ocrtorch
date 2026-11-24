"""Command-line interface for evaluating PP-OCRv6 PyTorch checkpoints on RxHandBD splits."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ocrtorch.cli.train import get_default_device
from ocrtorch.data.dataset import load_character_table, load_manifest
from ocrtorch.data.images import resize_rec_image
from ocrtorch.data.vocab import ctc_greedy_decode
from ocrtorch.domain.metrics import (
    character_error_rate,
    exact_match_accuracy,
    word_error_rate,
)
from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


def load_csv_rows(csv_path: Path, image_dir: Path) -> list[dict[str, str]]:
    """Load evaluation rows from a CSV file with Images and Text columns."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"Images", "Text"}.issubset(rows[0]):
        raise ValueError(f"Expected Images and Text columns in {csv_path}")
    examples: list[dict[str, str]] = []
    for row in rows:
        image_name = row["Images"].strip()
        text = row["Text"]
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing evaluation image: {image_path}")
        examples.append({"image": str(image_path), "text": text})
    return examples


class EvalOCRDataset(Dataset):
    """Deterministic evaluation dataset from a list of {image, text} records."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        with Image.open(row["image"]) as im:
            tensor_chw, _ = resize_rec_image(im.convert("RGB"))
        return {
            "image": torch.from_numpy(tensor_chw).to(torch.float32),
            "image_path": row["image"],
            "text": row["text"],
        }


def eval_collate_fn(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "image_paths": [item["image_path"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    inference_yaml: Path,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    predictions_csv: Path | None = None,
) -> dict:
    """Run deterministic inference on rows and compute corpus OCR metrics."""
    chars, _ = load_character_table(inference_yaml)
    num_classes = len(chars) + 1

    model = PP_OCRv6MediumRec(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    dataset = EvalOCRDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate_fn,
    )

    all_predictions: list[str] = []
    all_references: list[str] = []
    all_paths: list[str] = []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            logits = model(images)
            decoded = ctc_greedy_decode(logits.argmax(dim=-1), characters=chars)
            all_predictions.extend(decoded)
            all_references.extend(batch["texts"])
            all_paths.extend(batch["image_paths"])
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    raw_em, norm_em = exact_match_accuracy(all_references, all_predictions)
    cer = character_error_rate(all_references, all_predictions)
    wer = word_error_rate(all_references, all_predictions)

    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with predictions_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "reference", "prediction"])
            for path, ref, pred in zip(all_paths, all_references, all_predictions):
                writer.writerow([path, ref, pred])

    return {
        "checkpoint": str(checkpoint_path),
        "count": len(all_references),
        "raw_exact_match": raw_em,
        "norm_exact_match": norm_em,
        "cer": cer,
        "wer": wer,
        "word_accuracy": 1.0 - wer,
        "total_time_sec": elapsed,
        "throughput_img_per_sec": len(all_references) / max(elapsed, 1e-9),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--inference-yaml", type=Path, required=True, help="Path to Paddle inference.yml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="Path to JSONL manifest (train.jsonl / validation.jsonl)")
    group.add_argument("--csv", type=Path, help="Path to CSV labels (e.g. Test_Labels.csv)")
    parser.add_argument("--image-dir", type=Path, help="Image directory required when --csv is used")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "tpu", "mps", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--predictions-csv", type=Path, default=None, help="Optional output CSV path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.csv is not None:
        if args.image_dir is None:
            parser.error("--image-dir is required when --csv is specified")
        rows = load_csv_rows(args.csv, args.image_dir)
    else:
        rows = load_manifest(args.manifest)

    device = get_default_device(args.device)
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        inference_yaml=args.inference_yaml,
        rows=rows,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        predictions_csv=args.predictions_csv,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
