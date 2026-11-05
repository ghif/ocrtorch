"""Convert parameter arrays into PyTorch state_dict."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


def convert_arrays_to_pt(
    arrays_npz: Path | str,
    output_pt: Path | str,
) -> dict[str, torch.Tensor]:
    """Convert flat JAX/Paddle array checkpoint into PyTorch state_dict."""
    data = np.load(str(arrays_npz))
    model = PP_OCRv6MediumRec()
    state = model.state_dict()
    mapped = {}

    for k, v in state.items():
        if k.endswith(".num_batches_tracked"):
            continue
        parts = k.split(".")
        if parts[-1] == "weight":
            if parts[-2] in ("conv", "branch1", "branch3", "fc1", "fc2"):
                jax_path = "/".join(parts[:-1]) + "/kernel"
            elif parts[-2] in ("bn", "identity_bn"):
                jax_path = "/".join(parts[:-1]) + "/scale"
            elif parts[-2] in ("norm", "norm1", "norm2"):
                jax_path = "/".join(parts[:-1]) + "/scale"
            elif parts[-2] in ("qkv", "proj"):
                jax_path = "/".join(parts[:-1]) + "/kernel"
            elif parts[0] == "ctc":
                jax_path = "ctc/kernel"
            else:
                jax_path = "/".join(parts[:-1]) + "/kernel"
        elif parts[-1] == "bias":
            jax_path = "/".join(parts[:-1]) + "/bias"
        elif parts[-1] == "running_mean":
            jax_path = "/".join(parts[:-1]) + "/mean"
        elif parts[-1] == "running_var":
            jax_path = "/".join(parts[:-1]) + "/var"
        else:
            raise ValueError(f"Unknown param {k}")

        if jax_path not in data:
            raise KeyError(f"Missing parameter array for {jax_path}")

        val = data[jax_path]
        if val.ndim == 4:
            # HWIO -> OIHW
            val = np.transpose(val, (3, 2, 0, 1))
        elif val.ndim == 2:
            # IO -> OI
            val = val.T
        mapped[k] = torch.from_numpy(val)

    output_path = Path(output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, output_path)
    print(f"Successfully converted and saved {len(mapped)} tensors to {output_path}")
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Convert parameter arrays to PyTorch state dict")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .npz parameter arrays")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .pt checkpoint path")
    args = parser.parse_args()
    convert_arrays_to_pt(args.input, args.output)


if __name__ == "__main__":
    main()
