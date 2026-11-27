"""Image preprocessing compatible with PP-OCRv6 recognition inference."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def resize_rec_image(
    image: Image.Image | np.ndarray,
    image_height: int = 48,
    image_width: int = 320,
    max_image_width: int = 3200,
    dynamic_width: bool = False,
) -> tuple[np.ndarray, int]:
    """PP-OCR RecResizeImg behavior: BGR, fit-height, right-pad, [-1, 1], CHW."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        image = rgb[:, :, ::-1]  # PaddleOCR DecodeImage uses BGR.
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got {image.shape}")
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions {image.shape}")
    proportional_w = max(1, int(np.ceil(image_height * w / h)))
    target_width = min(max_image_width, max(image_width, proportional_w)) if dynamic_width else image_width
    resized_w = min(target_width, proportional_w)
    resized = Image.fromarray(image.astype(np.uint8)).resize(
        (resized_w, image_height), Image.Resampling.BILINEAR
    )
    resized = np.asarray(resized, dtype=np.float32)
    canvas = np.zeros((image_height, target_width, 3), dtype=np.float32)
    canvas[:, :resized_w] = resized
    canvas = canvas / 255.0
    canvas = (canvas - 0.5) / 0.5
    return np.transpose(canvas, (2, 0, 1)), resized_w


def to_torch_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    """Preprocess image and return a PyTorch tensor with shape [1, 3, 48, 320]."""
    chw, _ = resize_rec_image(image)
    return torch.from_numpy(chw).unsqueeze(0).to(dtype=torch.float32)
