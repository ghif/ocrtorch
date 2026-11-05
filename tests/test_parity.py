"""Numerical parity test verifying PyTorch outputs against reference activations."""

from pathlib import Path
import numpy as np
import pytest
import torch

from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


@pytest.fixture
def artifacts_dir():
    return Path(__file__).resolve().parent.parent / "artifacts"


def test_numerical_parity_against_reference(artifacts_dir):
    checkpoint_path = artifacts_dir / "ppocrv6_medium_rec.pt"
    ref_path = artifacts_dir / "ppocrv6_reference_activations.npz"

    assert checkpoint_path.exists(), f"Missing checkpoint {checkpoint_path}"
    assert ref_path.exists(), f"Missing reference file {ref_path}"

    state_dict = torch.load(checkpoint_path, weights_only=True)
    ref = np.load(ref_path)

    model = PP_OCRv6MediumRec()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_input = torch.from_numpy(ref["input"])
    with torch.no_grad():
        intermediates = model.forward_intermediates(test_input)

    stages = [
        "backbone_stem",
        "backbone_blocks2",
        "backbone_blocks3",
        "backbone_blocks4",
        "backbone_blocks5",
        "backbone_pool",
        "neck_reduce",
        "neck_local",
        "neck_local_sum",
        "neck_block_0",
        "neck_block_1",
        "neck_norm",
        "neck_out",
        "sequence",
        "ctc_logits",
        "probabilities",
    ]

    for stage in stages:
        actual = intermediates[stage].detach().cpu().numpy()
        expected = ref[stage]
        max_abs_err = np.max(np.abs(actual - expected))
        # Float32 precision across 50+ deep convolutional and attention layers
        assert max_abs_err < 1e-4, f"Stage {stage} exceeded tolerance: max_abs_err={max_abs_err}"

    # Verify exact greedy token argmax parity (100% agreement)
    argmax_actual = np.argmax(intermediates["probabilities"].numpy(), axis=-1)
    argmax_expected = np.argmax(ref["probabilities"], axis=-1)
    assert np.array_equal(argmax_actual, argmax_expected), "Argmax token mismatch found"
