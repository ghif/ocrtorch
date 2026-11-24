"""Unit tests for ocrtorch.cli.evaluate and word_error_rate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from ocrtorch.cli.evaluate import evaluate_checkpoint, load_csv_rows, main as evaluate_main
from ocrtorch.domain.metrics import word_error_rate
from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


def test_word_error_rate_basic() -> None:
    assert word_error_rate(["Hello World", "Tab 10mg"], ["hello  world", "Tab 20mg"]) == pytest.approx(0.25)
    assert word_error_rate(["", ""], ["", ""]) == 0.0


def test_evaluate_checkpoint_on_csv_and_jsonl(tmp_path: Path) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img_path = img_dir / "P0001.jpg"
    Image.new("RGB", (64, 32), color=(255, 255, 255)).save(img_path)

    csv_path = tmp_path / "Test_Labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Images", "Text"])
        writer.writerow(["P0001.jpg", "ab"])

    rows = load_csv_rows(csv_path, img_dir)
    assert len(rows) == 1
    assert rows[0]["text"] == "ab"

    yaml_path = tmp_path / "inference.yml"
    yaml_path.write_text(yaml.safe_dump({"PostProcess": {"character_dict": ["a", "b"]}}), encoding="utf-8")

    # 2 chars + space + blank = 4 classes
    model = PP_OCRv6MediumRec(num_classes=4)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    pred_csv = tmp_path / "preds.csv"
    result = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        inference_yaml=yaml_path,
        rows=rows,
        device=torch.device("cpu"),
        batch_size=1,
        num_workers=0,
        predictions_csv=pred_csv,
    )
    assert result["count"] == 1
    assert "raw_exact_match" in result
    assert "norm_exact_match" in result
    assert "cer" in result
    assert "wer" in result
    assert "word_accuracy" in result
    assert pred_csv.is_file()

    # Also test CLI main via JSONL manifest
    manifest_path = tmp_path / "val.jsonl"
    manifest_path.write_text(json.dumps({"image": str(img_path), "text": "ab"}) + "\n", encoding="utf-8")
    exit_code = evaluate_main([
        "--checkpoint", str(ckpt_path),
        "--inference-yaml", str(yaml_path),
        "--manifest", str(manifest_path),
        "--device", "cpu",
        "--num-workers", "0",
    ])
    assert exit_code == 0
