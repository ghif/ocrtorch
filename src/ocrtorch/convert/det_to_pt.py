"""Convert PP-OCRv6 medium detection .npz weights into a PyTorch state_dict."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ocrtorch.models.ppocrv6_det import PP_OCRv6MediumDet


def convert_det_npz_to_pt(
    arrays_npz: Path | str,
    output_pt: Path | str,
) -> dict[str, torch.Tensor]:
    """Convert exported PP-OCRv6 detection .npz arrays into a PyTorch state_dict."""
    data = np.load(str(arrays_npz))
    initial_params = {
        key: torch.from_numpy(np.array(data[key], dtype=np.float32))
        for key in data.files
    }
    model = PP_OCRv6MediumDet(initial_params=initial_params)
    state = model.state_dict()
    output_path = Path(output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(f"Successfully converted {len(state)} detection tensors to {output_path}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PP-OCRv6 detection .npz to PyTorch .pt")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input detection .npz file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .pt checkpoint path")
    args = parser.parse_args()
    convert_det_npz_to_pt(args.input, args.output)


if __name__ == "__main__":
    main()
