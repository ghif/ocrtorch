"""CLI for running the Buku KIA PP-OCRv6 detection + routed recognition pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from ocrtorch.cli.train import get_default_device
from ocrtorch.pipeline.engine import BukuKIAPipeline, RoutedOCRRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Buku KIA PP-OCRv6 end-to-end or crop-level OCR")
    parser.add_argument("--image", type=Path, required=True, help="Input full-page image or single crop image")
    parser.add_argument("--inference-yaml", type=Path, required=True, help="Path to Paddle inference.yml")
    parser.add_argument(
        "--det-checkpoint",
        type=Path,
        default=Path("artifacts/ppocrv6_medium_det.pt"),
        help="Path to PP-OCRv6 detection .pt checkpoint",
    )
    parser.add_argument(
        "--rec-checkpoint",
        type=Path,
        default=Path("artifacts/ppocrv6_medium_rec.pt"),
        help="Path to pretrained base recognition .pt checkpoint",
    )
    parser.add_argument(
        "--page3-checkpoint",
        type=Path,
        default=None,
        help="Optional fine-tuned recognition .pt checkpoint for page3 target fields",
    )
    parser.add_argument("--page-id", type=str, default=None, help="Page identifier (e.g. 'page3')")
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Field name for --crop-only mode (e.g. 'dates', 'blood_pressure', 'hemoglobin')",
    )
    parser.add_argument(
        "--crop-only",
        action="store_true",
        help="Skip detection and run recognition directly on --image as a single crop",
    )
    parser.add_argument(
        "--roi-json",
        type=Path,
        default=None,
        help="Optional JSON mapping field names to normalized [x1, y1, x2, y2] ROI boxes",
    )
    parser.add_argument(
        "--route-all-page3",
        action="store_true",
        help="Route all detected boxes on page3 to --page3-checkpoint when --roi-json is omitted",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "tpu", "mps", "cpu", "cuda"])
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON predictions")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = get_default_device(args.device)

    with Image.open(args.image) as im:
        pil_img = im.convert("RGB")

    if args.crop_only:
        recognizer = RoutedOCRRecognizer.from_checkpoints(
            base_checkpoint=args.rec_checkpoint,
            inference_yaml=args.inference_yaml,
            device=device,
            page3_checkpoint=args.page3_checkpoint,
        )
        texts, scores, routes = recognizer.recognize_crops(
            [pil_img],
            page_id=args.page_id,
            fields=[args.field],
        )
        payload = {
            "mode": "crop_only",
            "page_id": args.page_id,
            "field": args.field,
            "text": texts[0],
            "rec_score": scores[0],
            "model_route": routes[0],
        }
    else:
        roi_config = json.loads(args.roi_json.read_text(encoding="utf-8")) if args.roi_json else None
        pipeline = BukuKIAPipeline.from_checkpoints(
            det_checkpoint=args.det_checkpoint,
            base_rec_checkpoint=args.rec_checkpoint,
            inference_yaml=args.inference_yaml,
            page3_rec_checkpoint=args.page3_checkpoint,
            device=device,
        )
        result = pipeline.process_page(
            pil_img,
            page_id=args.page_id,
            roi_config=roi_config,
            route_all_page3=args.route_all_page3,
        )
        payload = {
            "mode": "full_page",
            "page_id": result.page_id,
            "t_det_ms": round(result.t_det_ms, 2),
            "t_rec_ms": round(result.t_rec_ms, 2),
            "t_total_ms": round(result.t_total_ms, 2),
            "full_text": result.full_text,
            "items": [
                {
                    "index": it.index,
                    "box": it.box.tolist() if it.box is not None else None,
                    "det_score": round(it.det_score, 4),
                    "text": it.text,
                    "rec_score": round(it.rec_score, 4),
                    "field": it.field,
                    "model_route": it.model_route,
                }
                for it in result.items
            ],
        }

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
