"""Tests for BukuKIAPipeline, RoutedOCRRecognizer, and ROI field routing."""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec
from ocrtorch.models.ppocrv6_det import PP_OCRv6MediumDet
from ocrtorch.pipeline.engine import BukuKIAPipeline, RoutedOCRRecognizer, match_box_to_roi


def _constant_rec_model(target_token_id: int, num_classes: int = 6) -> PP_OCRv6MediumRec:
    model = PP_OCRv6MediumRec(num_classes=num_classes)
    with torch.no_grad():
        model.ctc.weight.zero_()
        model.ctc.bias.fill_(-10.0)
        model.ctc.bias[target_token_id] = 10.0
    return model


def test_routed_recognizer_dispatches_page3_target_fields_only():
    chars = ["a", "b", "c", "d", "e"]
    base_model = _constant_rec_model(target_token_id=1, num_classes=6)  # predicts "a"
    page3_model = _constant_rec_model(target_token_id=2, num_classes=6)  # predicts "b"

    recognizer = RoutedOCRRecognizer(
        base_model=base_model,
        characters=chars,
        device="cpu",
        page3_model=page3_model,
    )

    crops = [Image.new("RGB", (64, 24), color=(255, 255, 255)) for _ in range(4)]

    # On page3: dates, blood_pressure, hemoglobin -> page3_finetuned; other/None -> base
    texts, scores, routes = recognizer.recognize_crops(
        crops,
        page_id="page3",
        fields=["dates", "blood_pressure", "hemoglobin", "patient_name"],
    )
    assert routes == ["page3_finetuned", "page3_finetuned", "page3_finetuned", "base"]
    assert texts == ["b", "b", "b", "a"]
    assert all(s > 0.9 for s in scores)

    # On page1: even if field is named "dates", it uses base
    texts_p1, _, routes_p1 = recognizer.recognize_crops(
        crops[:2],
        page_id="page1",
        fields=["dates", None],
    )
    assert routes_p1 == ["base", "base"]
    assert texts_p1 == ["a", "a"]


def test_match_box_to_roi():
    roi_cfg = {
        "dates": [0.05, 0.1, 0.3, 0.4],
        "hemoglobin": [[0.6, 0.2, 0.9, 0.35]],
    }
    box_date = np.array([[100, 200], [200, 200], [200, 250], [100, 250]], dtype=np.float32)
    box_hb = np.array([[700, 250], [850, 250], [850, 300], [700, 300]], dtype=np.float32)
    box_other = np.array([[400, 700], [500, 700], [500, 750], [400, 750]], dtype=np.float32)

    assert match_box_to_roi(box_date, (1000, 1000), roi_cfg) == "dates"
    assert match_box_to_roi(box_hb, (1000, 1000), roi_cfg) == "hemoglobin"
    assert match_box_to_roi(box_other, (1000, 1000), roi_cfg) is None


def test_bukukia_pipeline_process_page_smoke():
    det_model = PP_OCRv6MediumDet()
    rec_base = _constant_rec_model(target_token_id=1, num_classes=6)
    rec_p3 = _constant_rec_model(target_token_id=2, num_classes=6)
    recognizer = RoutedOCRRecognizer(
        base_model=rec_base,
        characters=["a", "b", "c", "d", "e"],
        device="cpu",
        page3_model=rec_p3,
    )
    pipeline = BukuKIAPipeline(det_model=det_model, recognizer=recognizer)

    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    result = pipeline.process_page(img, page_id="page3")
    assert result.page_id == "page3"
    assert result.original_size == (128, 128)
    assert result.t_det_ms >= 0.0
    assert result.t_total_ms >= 0.0
