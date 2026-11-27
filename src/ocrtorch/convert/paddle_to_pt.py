"""Convert original PaddleOCR PP-OCRv6 weights (or exported NPZ) into a PyTorch state_dict."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


def load_paddle_pir_params(model_dir: Path | str) -> dict[str, np.ndarray]:
    """Load parameter tensors directly from a Paddle 3.x PIR model directory."""
    model_path = Path(model_dir)
    json_file = model_path / "inference.json"
    params_file = model_path / "inference.pdiparams"
    if not json_file.is_file() or not params_file.is_file():
        raise FileNotFoundError(
            f"Expected inference.json and inference.pdiparams in {model_path}"
        )

    program = json.loads(json_file.read_text(encoding="utf-8"))
    ops = program["program"]["regions"][0]["blocks"][0]["ops"]
    # Paddle static save_vars serializes parameters sorted alphabetically by name.
    params = sorted(
        (op["A"][3], op["O"]["TT"]["D"][1]) for op in ops if op.get("#") == "p"
    )

    raw = params_file.read_bytes()
    offset = 0
    arrays: dict[str, np.ndarray] = {}
    for name, shape in params:
        _ver1, lod_level, _ver2, desc_size = struct.unpack_from("<IQII", raw, offset)
        offset += 20 + lod_level * 8 + desc_size
        numel = int(np.prod(shape))
        arr = np.frombuffer(raw, dtype="<f4", count=numel, offset=offset).reshape(shape)
        offset += numel * 4
        arrays[name] = arr.copy()

    if offset != len(raw):
        raise ValueError(
            f"Parsed {offset} bytes from {params_file}, expected {len(raw)} bytes"
        )
    return arrays


def convert_paddle_arrays_to_pt(
    arrays: dict[str, np.ndarray],
    output_pt: Path | str,
) -> dict[str, torch.Tensor]:
    """Map Paddle PIR PP-OCRv6 medium recognition arrays to PyTorch state_dict."""
    model = PP_OCRv6MediumRec()
    state = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def assign(key: str, value: np.ndarray) -> None:
        if key not in state:
            raise KeyError(f"PyTorch state_dict key does not exist: {key}")
        expected = tuple(state[key].shape)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != expected:
            raise ValueError(f"Shape mismatch for {key}: {arr.shape} != {expected}")
        mapped[key] = torch.from_numpy(arr.copy())

    def get(key: str) -> np.ndarray:
        if key not in arrays:
            raise KeyError(f"Missing Paddle tensor: {key}")
        used.add(key)
        return arrays[key]

    def assign_identity_bn(bn_prefix: str, channels: int) -> None:
        eps = 1e-5
        assign(f"{bn_prefix}.weight", np.ones(channels, dtype=np.float32))
        assign(f"{bn_prefix}.bias", np.zeros(channels, dtype=np.float32))
        assign(f"{bn_prefix}.running_mean", np.zeros(channels, dtype=np.float32))
        assign(f"{bn_prefix}.running_var", np.full(channels, 1.0 - eps, dtype=np.float32))

    def assign_fused_convbn(prefix: str, layer: int) -> None:
        w_key = f"conv2d_{layer}.w_0"
        b_key = f"conv2d_{layer}.b_0"
        kernel = get(w_key)  # Already OIHW in Paddle
        assign(f"{prefix}.conv.weight", kernel)
        channels = kernel.shape[0]
        bias = get(b_key) if b_key in arrays else np.zeros(channels, dtype=np.float32)
        assign(f"{prefix}.conv.bias", bias)
        assign_identity_bn(f"{prefix}.bn", channels)

    def assign_repdw(prefix: str, layer: int) -> None:
        w_key = f"conv2d_{layer}.w_0"
        b_key = f"conv2d_{layer}.b_0"
        kernel = get(w_key).copy()  # OIHW: [C, 1, 3, 3]
        if kernel.shape[1:] != (1, 3, 3):
            raise ValueError(f"Unexpected RepDW kernel shape: {kernel.shape}")
        # Paddle fuses the identity branch into the 3x3 kernel; subtract it so
        # RepDWConv's explicit identity_bn residual matches eval behavior.
        kernel[:, 0, 1, 1] -= 1.0
        channels = kernel.shape[0]
        assign(f"{prefix}.branch3.conv.weight", kernel)
        bias = get(b_key) if b_key in arrays else np.zeros(channels, dtype=np.float32)
        assign(f"{prefix}.branch3.conv.bias", bias)
        assign_identity_bn(f"{prefix}.branch3.bn", channels)
        assign(
            f"{prefix}.branch1.weight",
            np.zeros(state[f"{prefix}.branch1.weight"].shape, dtype=np.float32),
        )
        assign_identity_bn(f"{prefix}.identity_bn", channels)

    def assign_se(prefix: str, first: int, second: int) -> None:
        for name, layer in (("fc1", first), ("fc2", second)):
            assign(f"{prefix}.{name}.weight", get(f"conv2d_{layer}.w_0"))
            assign(f"{prefix}.{name}.bias", get(f"conv2d_{layer}.b_0"))

    # Backbone stem
    for stem_name, layer in (
        ("stem1", 73),
        ("stem2a", 74),
        ("stem2b", 75),
        ("stem3", 76),
        ("stem4", 77),
    ):
        assign_fused_convbn(f"backbone.conv1.{stem_name}.block", layer)

    # Backbone blocks: (stage, index, dw_layer, expand_layer, compress_layer, is_repdw, se_layers)
    blocks = [
        ("blocks2", 0, 79, 80, 81, True, (7, 8)),
        ("blocks3", 0, 82, 83, 84, False, None),
        ("blocks3", 1, 86, 87, 88, True, None),
        ("blocks3", 2, 90, 91, 92, True, (20, 21)),
        ("blocks4", 0, 93, 94, 95, False, None),
        ("blocks4", 1, 97, 98, 99, True, (29, 30)),
        ("blocks4", 2, 101, 102, 103, True, None),
        ("blocks4", 3, 105, 106, 107, True, (39, 40)),
        ("blocks4", 4, 109, 110, 111, True, None),
        ("blocks4", 5, 113, 114, 115, True, (49, 50)),
        ("blocks4", 6, 117, 118, 119, True, None),
        ("blocks5", 0, 120, 121, 122, False, None),
        ("blocks5", 1, 124, 125, 126, True, (62, 63)),
        ("blocks5", 2, 128, 129, 130, True, None),
    ]
    for stage, idx, dw_l, exp_l, cmp_l, is_repdw, se_layers in blocks:
        root = f"backbone.{stage}.{idx}"
        if is_repdw:
            assign_repdw(f"{root}.dw", dw_l)
        else:
            assign_fused_convbn(f"{root}.dw", dw_l)
        if se_layers is not None:
            assign_se(f"{root}.se", *se_layers)
        assign_fused_convbn(f"{root}.expand", exp_l)
        assign_fused_convbn(f"{root}.compress", cmp_l)

    # Neck convolutions (conv70/bn58, conv71/bn59, conv72/bn60)
    for name, conv_id, bn_id in (
        ("conv_reduce", 70, 58),
        ("local", 71, 59),
        ("skip_conv", 72, 60),
    ):
        root = f"neck.{name}.block"
        w = get(f"conv2d_{conv_id}.w_0")
        assign(f"{root}.conv.weight", w)
        assign(f"{root}.conv.bias", np.zeros(w.shape[0], dtype=np.float32))
        assign(f"{root}.bn.weight", get(f"batch_norm2d_{bn_id}.w_0"))
        assign(f"{root}.bn.bias", get(f"batch_norm2d_{bn_id}.b_0"))
        assign(f"{root}.bn.running_mean", get(f"batch_norm2d_{bn_id}.w_1"))
        assign(f"{root}.bn.running_var", get(f"batch_norm2d_{bn_id}.w_2"))

    # LightSVTR transformer blocks (Linear weights are IO in Paddle -> OI in PyTorch)
    for block in range(2):
        linear_base = block * 4
        norm_base = block * 2
        for name, layer in (
            ("qkv", linear_base),
            ("proj", linear_base + 1),
            ("fc1", linear_base + 2),
            ("fc2", linear_base + 3),
        ):
            assign(f"neck.blocks.{block}.{name}.weight", get(f"linear_{layer}.w_0").T)
            assign(f"neck.blocks.{block}.{name}.bias", get(f"linear_{layer}.b_0"))
        for name, layer in (("norm1", norm_base), ("norm2", norm_base + 1)):
            assign(f"neck.blocks.{block}.{name}.weight", get(f"layer_norm_{layer}.w_0"))
            assign(f"neck.blocks.{block}.{name}.bias", get(f"layer_norm_{layer}.b_0"))

    assign("neck.norm.weight", get("layer_norm_4.w_0"))
    assign("neck.norm.bias", get("layer_norm_4.b_0"))

    # CTC head
    assign("ctc.weight", get("linear_8.w_0").T)
    assign("ctc.bias", get("linear_8.b_0"))

    unmapped = sorted(set(arrays.keys()) - used - {"__metadata__"})
    if unmapped:
        raise ValueError(f"Unmapped Paddle tensors ({len(unmapped)}): {unmapped}")

    output_path = Path(output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, output_path)
    print(
        f"Successfully converted {len(used)} Paddle tensors into {len(mapped)} PyTorch tensors at {output_path}"
    )
    return mapped


def convert_arrays_to_pt(
    source: Path | str,
    output_pt: Path | str,
) -> dict[str, torch.Tensor]:
    """Convert original PaddleOCR model directory or Paddle PIR .npz weights into PyTorch state_dict."""
    src_path = Path(source)
    if src_path.is_dir():
        arrays = load_paddle_pir_params(src_path)
        return convert_paddle_arrays_to_pt(arrays, output_pt)

    with np.load(str(src_path)) as data:
        if "linear_8.w_0" not in data.files:
            raise ValueError(
                f"Expected Paddle PIR .npz or PaddleOCR model directory, got {src_path}"
            )
        arrays = {key: data[key] for key in data.files}
        return convert_paddle_arrays_to_pt(arrays, output_pt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert original PaddleOCR PP-OCRv6 model directory (or .npz) to PyTorch state_dict"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to PaddleOCR model directory (containing inference.json & inference.pdiparams) or .npz",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .pt checkpoint path",
    )
    args = parser.parse_args()
    convert_arrays_to_pt(args.input, args.output)


if __name__ == "__main__":
    main()
