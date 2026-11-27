"""PyTorch PP-OCRv6 medium text detector (DBNet++ inference path).

Executes the PP-OCRv6 medium detection graph in PyTorch with full
mathematical parity to the Paddle PIR / JAX NNX model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

GRAPH_FILE = Path(__file__).parent / "ppocrv6_det_graph.json"
SHAPES_FILE = Path(__file__).parent / "ppocrv6_det_shapes.json"


def _load_graph_plan() -> list[Any]:
    if not GRAPH_FILE.is_file():
        raise FileNotFoundError(f"Detection graph file not found: {GRAPH_FILE}")
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


def _sanitize_param_key(name: str) -> str:
    return name.replace(".", "__")


def _conv2d_with_padding(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: list[int] | tuple[int, int],
    padding: Any,
    dilation: list[int] | tuple[int, int],
    groups: int,
) -> torch.Tensor:
    sh, sw = int(stride[0]), int(stride[1])
    dh, dw = int(dilation[0]), int(dilation[1])
    if isinstance(padding, str):
        pad_upper = padding.upper()
        if pad_upper == "SAME":
            _, _, h, w_in = x.shape
            kh, kw = w.shape[2], w.shape[3]
            eff_kh = (kh - 1) * dh + 1
            eff_kw = (kw - 1) * dw + 1
            out_h = math.ceil(h / sh)
            out_w = math.ceil(w_in / sw)
            pad_h = max((out_h - 1) * sh + eff_kh - h, 0)
            pad_w = max((out_w - 1) * sw + eff_kw - w_in, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        pad = (0, 0)
    elif isinstance(padding[0], (list, tuple)):
        (pt, pb), (pl, pr) = padding[0], padding[1]
        if pt or pb or pl or pr:
            x = F.pad(x, (int(pl), int(pr), int(pt), int(pb)))
        pad = (0, 0)
    else:
        pad = (int(padding[0]), int(padding[1]))
    return F.conv2d(
        x,
        w,
        stride=(sh, sw),
        padding=pad,
        dilation=(dh, dw),
        groups=int(groups),
    )


def _max_pool2d_same(
    x: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
) -> torch.Tensor:
    _, _, h, w = x.shape
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    sh, sw = int(stride[0]), int(stride[1])
    out_h = math.ceil(h / sh)
    out_w = math.ceil(w / sw)
    pad_h = max((out_h - 1) * sh + kh - h, 0)
    pad_w = max((out_w - 1) * sw + kw - w, 0)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=float("-inf"),
        )
    return F.max_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw))


def _evaluate_plan(
    plan: list[Any],
    params: dict[str, torch.Tensor],
    image_nchw: torch.Tensor,
) -> torch.Tensor:
    """Execute the PyTorch detection forward pass."""
    env: dict[Any, Any] = {}
    for item in plan:
        tag = item[0]
        if tag == "p":
            env[item[1]] = params[item[2]]
        elif tag == "data":
            env[item[1]] = image_nchw
        elif tag == "const":
            env[item[1]] = item[2]
        elif tag == "full":
            env[item[1]] = torch.full(
                tuple(item[3]),
                float(item[2]),
                dtype=image_nchw.dtype,
                device=image_nchw.device,
            )
        elif tag == "conv":
            env[item[1]] = _conv2d_with_padding(
                env[item[2]],
                env[item[3]],
                stride=item[4],
                padding=item[5],
                dilation=item[6],
                groups=item[7],
            )
        elif tag == "conv_trans":
            env[item[1]] = F.conv_transpose2d(
                env[item[2]],
                env[item[3]],
                stride=(int(item[4][0]), int(item[4][1])),
                padding=(int(item[5][0]), int(item[5][1])),
                dilation=(int(item[6][0]), int(item[6][1])),
                groups=int(item[7]),
            )
        elif tag == "add":
            env[item[1]] = env[item[2]] + env[item[3]]
        elif tag == "mul":
            env[item[1]] = env[item[2]] * env[item[3]]
        elif tag == "relu":
            env[item[1]] = F.relu(env[item[2]])
        elif tag == "gelu":
            env[item[1]] = F.gelu(env[item[2]], approximate="none")
        elif tag == "hsigmoid":
            env[item[1]] = torch.clamp((env[item[2]] + 3.0) / 6.0, 0.0, 1.0)
        elif tag == "sigmoid":
            env[item[1]] = torch.sigmoid(env[item[2]])
        elif tag == "interp":
            x_in = env[item[2]]
            scale = item[3]
            if scale:
                sh = float(scale) if isinstance(scale, (int, float)) else float(scale[0])
                sw = float(scale) if isinstance(scale, (int, float)) else float(scale[1])
                env[item[1]] = F.interpolate(x_in, scale_factor=(sh, sw), mode="nearest")
            else:
                env[item[1]] = F.interpolate(x_in, size=(int(item[4]), int(item[5])), mode="nearest")
        elif tag == "pool":
            x_in = env[item[2]]
            if item[4] or item[3] == "avg":
                env[item[1]] = torch.mean(x_in, dim=(2, 3), keepdim=True)
            else:
                env[item[1]] = _max_pool2d_same(
                    x_in,
                    kernel_size=(int(item[5][0]), int(item[5][1])),
                    stride=(int(item[6][0]), int(item[6][1])),
                )
        elif tag == "mean":
            dims = tuple(item[3]) if isinstance(item[3], list) else item[3]
            env[item[1]] = torch.mean(env[item[2]], dim=dims, keepdim=bool(item[4]))
        elif tag == "bn":
            x_in = env[item[2]]
            mean = env[item[3]].reshape(1, -1, 1, 1)
            var = env[item[4]].reshape(1, -1, 1, 1)
            scale = env[item[5]].reshape(1, -1, 1, 1)
            bias = env[item[6]].reshape(1, -1, 1, 1)
            env[item[1]] = (x_in - mean) / torch.sqrt(var + float(item[7])) * scale + bias
        elif tag == "reshape":
            shape = env[item[3]] if item[3] is not None else item[4]
            env[item[1]] = torch.reshape(env[item[2]], tuple(shape))
        elif tag == "combine":
            env[item[1]] = [env[k] for k in item[2]]
        elif tag == "concat":
            env[item[1]] = torch.cat(env[item[2]], dim=int(item[3]))
        elif tag == "fetch":
            return env[item[1]]

    raise RuntimeError("Graph execution did not reach a fetch operation")


class PP_OCRv6MediumDet(nn.Module):
    """PP-OCRv6 medium text detection inference graph in PyTorch."""

    def __init__(self, initial_params: dict[str, torch.Tensor] | None = None):
        super().__init__()
        self._plan = _load_graph_plan()
        param_names = [item[2] for item in self._plan if item[0] == "p"]
        shapes = json.loads(SHAPES_FILE.read_text(encoding="utf-8")) if SHAPES_FILE.is_file() else {}

        self._orig_names: dict[str, str] = {}
        param_dict: dict[str, nn.Parameter] = {}
        for name in param_names:
            safe_name = _sanitize_param_key(name)
            self._orig_names[safe_name] = name
            if initial_params is not None and name in initial_params:
                tensor = initial_params[name].detach().to(torch.float32).clone()
            else:
                tensor = torch.zeros(tuple(shapes.get(name, (1,))), dtype=torch.float32)
            param_dict[safe_name] = nn.Parameter(tensor, requires_grad=False)

        self.params = nn.ParameterDict(param_dict)

    def forward(self, images_nchw: torch.Tensor) -> torch.Tensor:
        """Run text detection on normalized NCHW images and return (B, 1, H, W) prob map."""
        param_map = {
            self._orig_names[safe_name]: param
            for safe_name, param in self.params.items()
        }
        return _evaluate_plan(self._plan, param_map, images_nchw)
