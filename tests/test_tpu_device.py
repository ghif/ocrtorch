"""Tests for TorchTPU device resolution."""

import sys
from types import ModuleType
from unittest.mock import patch
import pytest
import torch

from ocrtorch.cli.train import get_default_device


def test_tpu_device_resolution_success():
    fake_torch_tpu = ModuleType("torch_tpu")
    with patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}):
        with patch.object(torch, "device", return_value="mock_tpu_device") as mock_device:
            dev = get_default_device("tpu")
            assert dev == "mock_tpu_device"
            mock_device.assert_called_once_with("tpu")


def test_tpu_device_resolution_failure():
    with patch.dict(sys.modules, {"torch_tpu": None}):
        with pytest.raises(RuntimeError, match="TorchTPU"):
            get_default_device("tpu")


def test_auto_device_selects_tpu_when_available():
    fake_torch_tpu = ModuleType("torch_tpu")
    with patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}):
        with patch.object(torch, "device", return_value="mock_tpu_device") as mock_device:
            dev = get_default_device("auto")
            assert dev == "mock_tpu_device"
            mock_device.assert_called_once_with("tpu")


def test_auto_device_falls_back_when_no_tpu():
    with patch.dict(sys.modules, {"torch_tpu": None}):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.object(torch.backends.mps, "is_available", return_value=False):
                dev = get_default_device("auto")
                assert dev == torch.device("cpu")
