"""Unit tests for Gemini AI Studio evaluation script."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from scripts import evaluate_gemini


def test_resolve_api_key_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Missing Gemini AI Studio API key"):
        evaluate_gemini.resolve_api_key()


def test_resolve_api_key_loads_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / ".env").write_text('GEMINI_API_KEY="test-secret-key"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert evaluate_gemini.resolve_api_key() == "test-secret-key"


def test_intermediate_persistence_latency_and_error_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("P1.jpg", "P2.jpg"):
        Image.new("RGB", (32, 16), color=(255, 255, 255)).save(img_dir / name)

    csv_path = tmp_path / "Test_Labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Images", "Text"])
        writer.writerow(["P1.jpg", "Nexcital"])
        writer.writerow(["P2.jpg", "Losita"])

    ckpt_jsonl = tmp_path / "intermediate.jsonl"
    # Pre-seed idx=0 as succeeded, idx=1 as errored (should be retried on resume)
    ckpt_jsonl.write_text(
        json.dumps({
            "idx": 0,
            "image": str(img_dir / "P1.jpg"),
            "reference": "Nexcital",
            "prediction": "Nexcital",
            "latency_ms": 120.0,
            "error": None,
        }) + "\n" + json.dumps({
            "idx": 1,
            "image": str(img_dir / "P2.jpg"),
            "reference": "Losita",
            "prediction": "",
            "latency_ms": None,
            "error": "RuntimeError: 403",
        }) + "\n",
        encoding="utf-8",
    )

    preds_csv = tmp_path / "preds.csv"
    summary_json = tmp_path / "summary.json"
    called_images: list[str] = []

    def fake_infer_one(_client, _model, image_path: str, _max_tokens: int = 1024) -> tuple[str, float]:
        called_images.append(Path(image_path).name)
        return "Losita", 240.0

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
    with (
        patch.object(evaluate_gemini.genai, "Client", return_value="fake-client") as mock_client,
        patch.object(evaluate_gemini, "infer_one", side_effect=fake_infer_one),
    ):
        rc = evaluate_gemini.main([
            "--csv", str(csv_path),
            "--image-dir", str(img_dir),
            "--workers", "1",
            "--log-every", "1",
            "--checkpoint-jsonl", str(ckpt_jsonl),
            "--predictions-csv", str(preds_csv),
            "--summary-json", str(summary_json),
        ])

    assert rc == 0
    mock_client.assert_called_once_with(vertexai=False, api_key="dummy-key")
    # Only P2.jpg should be retried since P1.jpg succeeded
    assert called_images == ["P2.jpg"]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["count"] == 2
    assert summary["norm_exact_match"] == 1.0
    assert summary["cer"] == 0.0
    assert summary["mean_latency_ms"] == 180.0
    assert summary["p50_latency_ms"] == 180.0
    assert summary["p95_latency_ms"] == 234.0
    assert summary["throughput_img_per_sec"] > 0

    with preds_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["latency_ms"] == "120.0"
    assert rows[1]["latency_ms"] == "240.0"
