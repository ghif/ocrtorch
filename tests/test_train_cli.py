"""Tests for training CLI config loading and TPU hybrid defaults."""

from __future__ import annotations

import pytest
import torch

from ocrtorch.cli.train import parse_args, resolve_runtime_options


def _base_cli_args() -> list[str]:
    return [
        "--inference-yaml", "inference.yml",
        "--train-manifest", "train.jsonl",
        "--val-manifest", "val.jsonl",
    ]


def test_parse_args_loads_yaml_and_cli_overrides(tmp_path):
    cfg_path = tmp_path / "tpu.yaml"
    cfg_path.write_text(
        "device: tpu\nbf16: true\nepochs: 40\nbatch_size: 64\nlr: 4.0e-5\n",
        encoding="utf-8",
    )
    args = parse_args([
        "--config", str(cfg_path),
        *_base_cli_args(),
        "--epochs", "12",
        "--no-bf16",
    ])
    assert args.device == "tpu"
    assert args.batch_size == 64
    assert args.lr == pytest.approx(4e-5)
    assert args.epochs == 12
    assert args.bf16 is False


def test_parse_args_rejects_unknown_yaml_keys(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("unknown_hyperparam: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key"):
        parse_args(["--config", str(cfg_path), *_base_cli_args()])


def test_resolve_runtime_options_defaults_hybrid_bf16_on_tpu():
    from types import SimpleNamespace

    args = parse_args(_base_cli_args())
    tpu_opts = resolve_runtime_options(args, SimpleNamespace(type="tpu"))
    assert tpu_opts["amp_dtype"] == torch.bfloat16
    assert tpu_opts["max_target_len"] == 40
    assert tpu_opts["drop_last"] is True

    cpu_opts = resolve_runtime_options(args, torch.device("cpu"))
    assert cpu_opts["amp_dtype"] is None
    assert cpu_opts["max_target_len"] is None
    assert cpu_opts["drop_last"] is False


def test_anti_forgetting_cli_flags_parse_cleanly():
    args = parse_args([
        *_base_cli_args(),
        "--backbone-lr-scale", "0.1",
        "--l2-sp-weight", "5e-4",
        "--protect-unseen-ctc",
        "--balance-fields",
        "--balance-alpha", "0.5",
    ])
    assert args.backbone_lr_scale == pytest.approx(0.1)
    assert args.l2_sp_weight == pytest.approx(5e-4)
    assert args.protect_unseen_ctc is True
    assert args.balance_fields is True
    assert args.balance_alpha == pytest.approx(0.5)
