"""Evaluate Gemini-3.8-Flash zero-shot OCR baseline on RxHandBD splits via Gemini AI Studio."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from google import genai
from google.genai import types

from ocrtorch.cli.evaluate import load_csv_rows
from ocrtorch.data.dataset import load_manifest
from ocrtorch.domain.metrics import (
    character_error_rate,
    exact_match_accuracy,
    word_error_rate,
)

PROMPT = (
    "Transcribe the exact handwritten text shown in this word crop image. "
    "Return only the transcribed text itself, with no quotes, markdown, or explanation."
)


def _load_dotenv_if_needed(paths: Sequence[Path]) -> None:
    """Populate GEMINI_API_KEY / GOOGLE_API_KEY from .env files if not set."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    for env_path in paths:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, _, val = line.partition("=")
            key = key.strip()
            if key not in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                continue
            val = val.strip()
            if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                val = val[1:-1]
            if val and not os.environ.get(key):
                os.environ[key] = val


def resolve_api_key(cli_api_key: str | None = None) -> str:
    """Resolve Gemini AI Studio API key from CLI, environment, or .env."""
    if cli_api_key:
        return cli_api_key
    _load_dotenv_if_needed((Path(".env"), Path("ocrtorch/.env")))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Gemini AI Studio API key. Set GEMINI_API_KEY (in env or .env) "
            "or pass --api-key."
        )
    return api_key


def infer_one(
    client: genai.Client,
    model: str,
    image_path: str,
    max_output_tokens: int = 1024,
    retries: int = 4,
) -> tuple[str, float]:
    """Run single-image OCR inference and return (prediction, latency_ms)."""
    img_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req_start = time.perf_counter()
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            latency_ms = (time.perf_counter() - req_start) * 1000.0
            text = (resp.text or "").strip()
            if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
                text = text[1:-1].strip()
            return text, round(latency_ms, 2)
        except Exception as exc:
            last_exc = exc
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"Failed inference on {image_path}: {last_exc}")


def load_intermediate_results(checkpoint_jsonl: Path, rows: list[dict[str, str]]) -> dict[int, dict]:
    """Load previously succeeded records from a JSONL checkpoint for resume."""
    completed: dict[int, dict] = {}
    if not checkpoint_jsonl.is_file():
        return completed
    for line in checkpoint_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = rec.get("idx")
        if (
            isinstance(idx, int)
            and 0 <= idx < len(rows)
            and rec.get("image") == rows[idx]["image"]
            and rec.get("error") is None
        ):
            completed[idx] = rec
    return completed


def compute_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    raw_em, norm_em = exact_match_accuracy(references, predictions)
    cer = character_error_rate(references, predictions)
    wer = word_error_rate(references, predictions)
    return {
        "raw_exact_match": raw_em,
        "norm_exact_match": norm_em,
        "cer": cer,
        "wer": wer,
        "word_accuracy": 1.0 - wer,
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_timing_stats(
    records: list[dict],
    new_count: int,
    elapsed_sec: float,
) -> dict[str, float]:
    """Compute per-request latency percentiles and session throughput."""
    latencies = sorted(
        float(r["latency_ms"])
        for r in records
        if r.get("error") is None and r.get("latency_ms") is not None
    )
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "mean_latency_ms": round(mean_lat, 2),
        "min_latency_ms": round(latencies[0], 2) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 50.0), 2),
        "p95_latency_ms": round(_percentile(latencies, 95.0), 2),
        "p99_latency_ms": round(_percentile(latencies, 99.0), 2),
        "total_time_sec": round(elapsed_sec, 3),
        "throughput_img_per_sec": round(new_count / max(elapsed_sec, 1e-9), 3) if new_count > 0 else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--api-key", default=None, help="Gemini AI Studio API key (defaults to GEMINI_API_KEY)")
    parser.add_argument("--csv", type=Path, default=Path("/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD/Test_Labels.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD/Test_Set"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=50, help="Persist rolling metrics every N completed items")
    parser.add_argument("--checkpoint-jsonl", type=Path, default=Path("ocrtorch/artifacts/gemini_3_8_flash_test_intermediate.jsonl"))
    parser.add_argument("--predictions-csv", type=Path, default=Path("ocrtorch/artifacts/gemini_3_8_flash_test_predictions.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("ocrtorch/artifacts/gemini_3_8_flash_eval.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_manifest(args.manifest) if args.manifest else load_csv_rows(args.csv, args.image_dir)
    api_key = resolve_api_key(args.api_key)
    client = genai.Client(vertexai=False, api_key=api_key)

    completed_records = load_intermediate_results(args.checkpoint_jsonl, rows)
    if completed_records:
        print(f"Resuming from {len(completed_records)}/{len(rows)} cached results in {args.checkpoint_jsonl}", flush=True)

    args.checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    pending_indices = [idx for idx in range(len(rows)) if idx not in completed_records]
    errors = 0
    new_completed = 0

    t0 = time.time()
    if pending_indices:
        with args.checkpoint_jsonl.open("a", encoding="utf-8") as ckpt_file:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_to_idx = {
                    pool.submit(infer_one, client, args.model, rows[idx]["image"], args.max_output_tokens): idx
                    for idx in pending_indices
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    err_msg: str | None = None
                    latency_ms: float | None = None
                    try:
                        pred, latency_ms = fut.result()
                    except Exception as exc:
                        errors += 1
                        pred = ""
                        err_msg = f"{type(exc).__name__}: {exc}"
                        print(f"[Warning] idx={idx} error: {err_msg}", flush=True)

                    rec = {
                        "idx": idx,
                        "image": rows[idx]["image"],
                        "reference": rows[idx]["text"],
                        "prediction": pred,
                        "latency_ms": latency_ms,
                        "error": err_msg,
                    }
                    completed_records[idx] = rec
                    new_completed += 1
                    ckpt_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ckpt_file.flush()

                    done = len(completed_records)
                    if done % args.log_every == 0 or done == len(rows):
                        ordered_indices = sorted(completed_records)
                        ordered_recs = [completed_records[i] for i in ordered_indices]
                        refs_so_far = [r["reference"] for r in ordered_recs]
                        preds_so_far = [r["prediction"] for r in ordered_recs]
                        m = compute_metrics(refs_so_far, preds_so_far)
                        timing = compute_timing_stats(ordered_recs, new_completed, time.time() - t0)
                        progress_summary = {
                            "status": "in_progress" if done < len(rows) else "complete",
                            "model": args.model,
                            "completed": done,
                            "total": len(rows),
                            "errors": errors,
                            **m,
                            **timing,
                        }
                        args.summary_json.write_text(json.dumps(progress_summary, indent=2) + "\n", encoding="utf-8")
                        print(
                            f"Completed {done}/{len(rows)} ({done / len(rows) * 100:.1f}%) | "
                            f"CER: {m['cer']*100:.2f}% | Norm EM: {m['norm_exact_match']*100:.2f}% | "
                            f"p50: {timing['p50_latency_ms']:.0f}ms | {timing['throughput_img_per_sec']:.2f} img/s",
                            flush=True,
                        )

    elapsed = time.time() - t0
    ordered_recs = [completed_records[i] for i in range(len(rows))]
    predictions = [r["prediction"] for r in ordered_recs]
    references = [row["text"] for row in rows]
    metrics = compute_metrics(references, predictions)
    timing = compute_timing_stats(ordered_recs, new_completed, elapsed)

    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "reference", "prediction", "latency_ms"])
        for row, rec in zip(rows, ordered_recs):
            writer.writerow([row["image"], row["text"], rec["prediction"], rec.get("latency_ms", "")])

    summary = {
        "status": "complete",
        "model": args.model,
        "count": len(rows),
        "errors": errors,
        **metrics,
        **timing,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
