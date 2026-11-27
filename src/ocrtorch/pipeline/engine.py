"""End-to-end Buku KIA PP-OCRv6 pipeline with page/field-routed recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ocrtorch.data.dataset import load_character_table
from ocrtorch.data.images import resize_rec_image
from ocrtorch.data.vocab import ctc_greedy_decode
from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec
from ocrtorch.models.ppocrv6_det import PP_OCRv6MediumDet
from ocrtorch.pipeline.db_postprocess import extract_boxes_from_prob_map, get_rotate_crop_image
from ocrtorch.pipeline.reading_order import sort_reading_order

PAGE3_TARGET_FIELDS = frozenset({"dates", "blood_pressure", "hemoglobin"})


@dataclass
class OCRItem:
    index: int
    box: np.ndarray | None
    det_score: float
    text: str
    rec_score: float
    field: str | None
    model_route: str


@dataclass
class BukuKIAPageResult:
    page_id: str | None
    items: list[OCRItem]
    full_text: str
    t_det_ms: float
    t_rec_ms: float
    t_total_ms: float
    original_size: tuple[int, int]


def resize_image_det(
    img: np.ndarray,
    max_side_len: int = 960,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize RGB image so max side <= max_side_len and dimensions are multiples of 32."""
    h, w = img.shape[:2]
    ratio = float(max_side_len) / max(h, w) if max(h, w) > max_side_len else 1.0
    resize_h = max(int(round((h * ratio) / 32) * 32), 32)
    resize_w = max(int(round((w * ratio) / 32) * 32), 32)
    resized_img = cv2.resize(img, (resize_w, resize_h))
    return resized_img, (h, w)


def normalize_det_image(img: np.ndarray) -> np.ndarray:
    """Normalize RGB uint8 image with ImageNet mean/std into [1, 3, H, W] float32."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    norm = (img.astype(np.float32) / 255.0 - mean) / std
    return np.transpose(norm, (2, 0, 1))[np.newaxis, ...]


def match_box_to_roi(
    box: np.ndarray,
    image_size: tuple[int, int],
    roi_config: dict[str, list[float] | list[list[float]]],
) -> str | None:
    """Match a 4x2 polygon box center against normalized [x1, y1, x2, y2] ROI regions."""
    orig_w, orig_h = image_size
    cx = float(box[:, 0].mean()) / max(1.0, float(orig_w))
    cy = float(box[:, 1].mean()) / max(1.0, float(orig_h))

    for field_name, regions in roi_config.items():
        boxes_list = [regions] if regions and isinstance(regions[0], (int, float)) else regions
        for region in boxes_list:
            x1, y1, x2, y2 = region  # type: ignore[misc]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return field_name
    return None


def _decode_with_scores(logits: torch.Tensor, characters: list[str]) -> tuple[list[str], list[float]]:
    """Greedy-decode CTC logits [B, T, V] and compute mean max-probability confidence."""
    probs = F.softmax(logits.float(), dim=-1)
    max_probs, pred_ids = probs.max(dim=-1)
    texts = ctc_greedy_decode(pred_ids, characters=characters)
    scores: list[float] = []
    for i in range(pred_ids.shape[0]):
        non_blank = pred_ids[i] != 0
        if non_blank.any():
            scores.append(float(max_probs[i][non_blank].mean().item()))
        else:
            scores.append(float(max_probs[i].mean().item()))
    return texts, scores


class RoutedOCRRecognizer:
    """Dual-checkpoint recognizer routing page3 target fields to the fine-tuned model."""

    def __init__(
        self,
        base_model: PP_OCRv6MediumRec,
        characters: list[str],
        device: torch.device | str,
        page3_model: PP_OCRv6MediumRec | None = None,
        target_fields: frozenset[str] = PAGE3_TARGET_FIELDS,
    ):
        self.device = torch.device(device)
        self.characters = characters
        self.base_model = base_model.to(self.device).eval()
        self.page3_model = page3_model.to(self.device).eval() if page3_model is not None else None
        self.target_fields = target_fields

    @classmethod
    def from_checkpoints(
        cls,
        base_checkpoint: Path | str,
        inference_yaml: Path | str,
        device: torch.device | str = "cpu",
        page3_checkpoint: Path | str | None = None,
        target_fields: frozenset[str] = PAGE3_TARGET_FIELDS,
    ) -> "RoutedOCRRecognizer":
        chars, _ = load_character_table(inference_yaml)
        num_classes = len(chars) + 1

        base_model = PP_OCRv6MediumRec(num_classes=num_classes)
        base_state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        base_model.load_state_dict(base_state, strict=False)

        page3_model = None
        if page3_checkpoint is not None:
            page3_model = PP_OCRv6MediumRec(num_classes=num_classes)
            p3_state = torch.load(page3_checkpoint, map_location="cpu", weights_only=True)
            page3_model.load_state_dict(p3_state, strict=False)

        return cls(
            base_model=base_model,
            characters=chars,
            device=device,
            page3_model=page3_model,
            target_fields=target_fields,
        )

    def select_route(self, page_id: str | None, field: str | None) -> str:
        """Select 'page3_finetuned' for page3 target fields, else 'base'."""
        if (
            self.page3_model is not None
            and page_id == "page3"
            and (field in self.target_fields or field == "__all__")
        ):
            return "page3_finetuned"
        return "base"

    @torch.no_grad()
    def recognize_crops(
        self,
        crops: list[Image.Image | np.ndarray],
        page_id: str | None = None,
        fields: list[str | None] | None = None,
        batch_size: int = 16,
    ) -> tuple[list[str], list[float], list[str]]:
        """Recognize cropped images, routing each crop to base or page3_finetuned."""
        if not crops:
            return [], [], []
        if fields is None:
            fields = [None] * len(crops)
        if len(fields) != len(crops):
            raise ValueError(f"fields length ({len(fields)}) must match crops length ({len(crops)})")

        routes = [self.select_route(page_id, f) for f in fields]
        texts = [""] * len(crops)
        scores = [0.0] * len(crops)

        for route_name, model in (("base", self.base_model), ("page3_finetuned", self.page3_model)):
            if model is None:
                continue
            idxs = [i for i, r in enumerate(routes) if r == route_name]
            if not idxs:
                continue
            # Sort by crop aspect ratio so dynamic-width batching minimizes right-padding.
            def _aspect_ratio(i: int) -> float:
                crop = crops[i]
                if isinstance(crop, Image.Image):
                    cw, ch = crop.size
                else:
                    ch, cw = crop.shape[:2]
                return cw / max(1.0, float(ch))

            idxs.sort(key=_aspect_ratio)
            for start in range(0, len(idxs), batch_size):
                chunk_idxs = idxs[start : start + batch_size]
                tensors = []
                for idx in chunk_idxs:
                    crop = crops[idx]
                    pil_crop = crop.convert("RGB") if isinstance(crop, Image.Image) else Image.fromarray(crop).convert("RGB")
                    chw, _ = resize_rec_image(pil_crop, dynamic_width=True)
                    tensors.append(torch.from_numpy(chw).to(torch.float32))
                max_w = max(t.shape[2] for t in tensors)
                if any(t.shape[2] != max_w for t in tensors):
                    tensors = [F.pad(t, (0, max_w - t.shape[2]), value=0.0) for t in tensors]
                batch_t = torch.stack(tensors, dim=0).to(self.device)
                logits = model(batch_t)
                chunk_texts, chunk_scores = _decode_with_scores(logits, self.characters)
                for idx, txt, sc in zip(chunk_idxs, chunk_texts, chunk_scores):
                    texts[idx] = txt
                    scores[idx] = sc

        return texts, scores, routes


class BukuKIAPipeline:
    """Full PyTorch PP-OCRv6 detection + routed recognition pipeline for Buku KIA."""

    def __init__(
        self,
        det_model: PP_OCRv6MediumDet,
        recognizer: RoutedOCRRecognizer,
    ):
        self.device = recognizer.device
        self.det_model = det_model.to(self.device).eval()
        self.recognizer = recognizer

    @classmethod
    def from_checkpoints(
        cls,
        det_checkpoint: Path | str,
        base_rec_checkpoint: Path | str,
        inference_yaml: Path | str,
        page3_rec_checkpoint: Path | str | None = None,
        device: torch.device | str = "cpu",
    ) -> "BukuKIAPipeline":
        det_model = PP_OCRv6MediumDet()
        det_state = torch.load(det_checkpoint, map_location="cpu", weights_only=True)
        det_model.load_state_dict(det_state, strict=True)
        recognizer = RoutedOCRRecognizer.from_checkpoints(
            base_checkpoint=base_rec_checkpoint,
            inference_yaml=inference_yaml,
            device=device,
            page3_checkpoint=page3_rec_checkpoint,
        )
        return cls(det_model=det_model, recognizer=recognizer)

    @torch.no_grad()
    def process_page(
        self,
        image_input: Image.Image | np.ndarray,
        page_id: str | None = None,
        roi_config: dict[str, list[float] | list[list[float]]] | None = None,
        field_router: Callable[[str | None, np.ndarray, Image.Image], str | None] | None = None,
        route_all_page3: bool = False,
        box_thresh: float = 0.45,
        det_thresh: float = 0.2,
        unclip_ratio: float = 1.4,
        max_side_len: int = 960,
        batch_size: int = 16,
    ) -> BukuKIAPageResult:
        """Run full-page detection, cropping, reading-order sorting, and routed recognition."""
        t_start = time.perf_counter()

        if isinstance(image_input, Image.Image):
            rgb_np = np.array(image_input.convert("RGB"))
        else:
            rgb_np = image_input if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)

        orig_h, orig_w = rgb_np.shape[:2]
        bgr_np = rgb_np[:, :, ::-1]
        resized_img, orig_shape = resize_image_det(bgr_np, max_side_len=max_side_len)
        norm_det = torch.from_numpy(normalize_det_image(resized_img)).to(self.device)

        t0_det = time.perf_counter()
        prob_map = self.det_model(norm_det)[0, 0].detach().cpu().numpy()
        t_det_ms = (time.perf_counter() - t0_det) * 1000.0

        boxes, det_scores = extract_boxes_from_prob_map(
            prob_map,
            orig_shape,
            thresh=det_thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )
        sorted_boxes, sorted_scores, _ = sort_reading_order(boxes, det_scores)
        sorted_scores = sorted_scores or [1.0] * len(sorted_boxes)

        crops: list[Image.Image] = []
        fields: list[str | None] = []
        for box in sorted_boxes:
            crop_np = get_rotate_crop_image(rgb_np, box)
            crop_pil = Image.fromarray(crop_np)
            crops.append(crop_pil)
            field_name: str | None = None
            if field_router is not None:
                field_name = field_router(page_id, box, crop_pil)
            elif roi_config is not None:
                field_name = match_box_to_roi(box, (orig_w, orig_h), roi_config)
            elif route_all_page3 and page_id == "page3":
                field_name = "__all__"
            fields.append(field_name)

        t0_rec = time.perf_counter()
        texts, rec_scores, routes = self.recognizer.recognize_crops(
            crops,
            page_id=page_id,
            fields=fields,
            batch_size=batch_size,
        )
        t_rec_ms = (time.perf_counter() - t0_rec) * 1000.0

        items = [
            OCRItem(
                index=i,
                box=box,
                det_score=float(d_sc),
                text=txt,
                rec_score=float(r_sc),
                field=fld,
                model_route=rt,
            )
            for i, (box, d_sc, txt, r_sc, fld, rt) in enumerate(
                zip(sorted_boxes, sorted_scores, texts, rec_scores, fields, routes)
            )
        ]
        full_text = "\n".join(it.text for it in items if it.text)
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        return BukuKIAPageResult(
            page_id=page_id,
            items=items,
            full_text=full_text,
            t_det_ms=t_det_ms,
            t_rec_ms=t_rec_ms,
            t_total_ms=t_total_ms,
            original_size=(orig_w, orig_h),
        )

    def recognize_crops(
        self,
        crops: list[Image.Image | np.ndarray],
        page_id: str = "page3",
        fields: list[str | None] | None = None,
        batch_size: int = 16,
    ) -> tuple[list[str], list[float], list[str]]:
        """Directly recognize pre-cropped fields without running detection."""
        return self.recognizer.recognize_crops(crops, page_id=page_id, fields=fields, batch_size=batch_size)
