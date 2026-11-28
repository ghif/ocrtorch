"""Command-line interface for PP-OCRv6 fine-tuning."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import DataLoader
import yaml

from ocrtorch.data.dataset import (
    OCRDataset,
    build_field_balanced_sampler,
    collect_active_token_ids,
    load_character_table,
    ocr_collate_fn,
)
from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec
from ocrtorch.train.trainer import OCRTrainer


def get_default_device(requested_device: str) -> torch.device:
    """Resolve compute device based on availability and request."""
    if requested_device == "tpu":
        try:
            import torch_tpu  # noqa: F401
            return torch.device("tpu")
        except Exception as exc:
            raise RuntimeError(
                "TorchTPU ('torch_tpu') is not available or TPU device could not be initialized. "
                "Ensure torch-tpu is installed in your TPU environment."
            ) from exc
    elif requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but Apple Silicon MPS is not available.")
        return torch.device("mps")
    elif requested_device == "cpu":
        return torch.device("cpu")
    elif requested_device == "auto":
        try:
            import torch_tpu  # noqa: F401
            return torch.device("tpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for PP-OCRv6 fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv6 with PyTorch on CPU, MPS, or TPU")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file (CLI flags override)")
    parser.add_argument("--checkpoint", type=str, default="artifacts/ppocrv6_medium_rec.pt", help="Pretrained .pt checkpoint")
    parser.add_argument("--inference-yaml", type=str, required=True, help="Path to Paddle inference.yml for character dict")
    parser.add_argument("--train-manifest", type=str, required=True, help="Path to train.jsonl manifest")
    parser.add_argument("--val-manifest", type=str, required=True, help="Path to validation.jsonl manifest")
    parser.add_argument("--output-dir", type=str, default="artifacts/checkpoints/finetuned", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "tpu", "mps", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in epochs")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--compile-backend",
        type=str,
        default=None,
        help="Optional torch.compile backend (e.g. 'tpu' or 'inductor'; disabled by default)",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use hybrid bfloat16 autocast with float32 master weights (defaults to enabled on TPU)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear LR warmup epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum cosine-annealed learning rate")
    parser.add_argument(
        "--max-target-len",
        type=int,
        default=None,
        help="Pad targets to fixed static length (defaults to 40 on TPU; 0 disables)",
    )
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop trailing partial batch in train DataLoader (defaults to enabled on TPU)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze PPLCNetV4Rec backbone weights during fine-tuning",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help="Learning rate multiplier for backbone parameters when unfrozen (e.g. 0.1)",
    )
    parser.add_argument(
        "--l2-sp-weight",
        type=float,
        default=0.0,
        help="L2-SP regularization weight anchoring parameters to pretrained checkpoint",
    )
    parser.add_argument(
        "--protect-unseen-ctc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero CTC head gradients for character IDs absent from the training manifest",
    )
    parser.add_argument(
        "--balance-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use field-balanced WeightedRandomSampler when manifest rows include a 'field' key",
    )
    parser.add_argument(
        "--balance-alpha",
        type=float,
        default=0.5,
        help="Inverse-frequency exponent for field-balanced sampling (0=uniform, 1=full inverse)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with optional YAML config defaults (CLI overrides YAML)."""
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    if known.config:
        cfg_path = Path(known.config)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {cfg_path}")
        valid_dests = {action.dest for action in parser._actions} - {"help", "config"}
        unknown_keys = sorted(set(raw_cfg.keys()) - valid_dests)
        if unknown_keys:
            raise ValueError(f"Unknown config key(s) in {cfg_path}: {', '.join(unknown_keys)}")
        parser.set_defaults(**raw_cfg)
    return parser.parse_args(argv)


def resolve_runtime_options(args: argparse.Namespace, device: torch.device) -> dict:
    """Resolve backend-dependent runtime options (hybrid bf16, static shapes, drop_last)."""
    is_tpu = device.type == "tpu"
    compile_backend = args.compile_backend if (args.compile_backend and args.compile_backend.lower() != "none") else None
    use_bf16 = args.bf16 if args.bf16 is not None else is_tpu
    amp_dtype = torch.bfloat16 if use_bf16 else None
    max_target_len = args.max_target_len if args.max_target_len is not None else (40 if is_tpu else None)
    if max_target_len == 0:
        max_target_len = None
    drop_last = args.drop_last if args.drop_last is not None else is_tpu
    return {
        "compile_backend": compile_backend,
        "amp_dtype": amp_dtype,
        "max_target_len": max_target_len,
        "drop_last": drop_last,
    }


def main():
    args = parse_args()
    device = get_default_device(args.device)
    opts = resolve_runtime_options(args, device)
    compile_backend = opts["compile_backend"]
    amp_dtype = opts["amp_dtype"]
    max_target_len = opts["max_target_len"]
    drop_last = opts["drop_last"]

    print(
        f"Using device: {device} | compile_backend={compile_backend} | "
        f"amp_dtype={amp_dtype} | max_target_len={max_target_len} | drop_last={drop_last}"
    )

    # Load character dictionary
    print(f"Loading vocabulary from {args.inference_yaml}...")
    chars, char_to_id = load_character_table(args.inference_yaml)
    num_classes = len(chars) + 1
    print(f"Vocabulary loaded: {len(chars)} characters ({num_classes} total classes including blank)")

    # Datasets and Loaders
    print("Preparing train and validation datasets...")
    train_dataset = OCRDataset(args.train_manifest, char_to_id, augment=True)
    val_dataset = OCRDataset(args.val_manifest, char_to_id, augment=False)
    print(f"Dataset sizes: {len(train_dataset)} training crops, {len(val_dataset)} validation crops")

    collate_fn = partial(ocr_collate_fn, max_target_len=max_target_len) if max_target_len else ocr_collate_fn
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    sampler = None
    shuffle = True
    if args.balance_fields:
        sampler = build_field_balanced_sampler(train_dataset.rows, alpha=args.balance_alpha)
        shuffle = False
        print(f"Field-balanced sampling enabled (alpha={args.balance_alpha})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Initialize model
    print("Initializing PP-OCRv6 recognizer...")
    model = PP_OCRv6MediumRec(num_classes=num_classes)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading pretrained weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Warning] Checkpoint {ckpt_path} not found, initializing from scratch.")

    active_token_ids = (
        collect_active_token_ids(train_dataset.rows, char_to_id, include_blank=True)
        if args.protect_unseen_ctc
        else None
    )
    if active_token_ids is not None:
        print(f"Protecting unseen CTC head rows: {len(active_token_ids)}/{num_classes} active token IDs")

    trainer = OCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        characters=chars,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        patience=args.patience,
        compile_backend=compile_backend,
        amp_dtype=amp_dtype,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        freeze_backbone=args.freeze_backbone,
        backbone_lr_scale=args.backbone_lr_scale,
        l2_sp_weight=args.l2_sp_weight,
        active_token_ids=active_token_ids,
    )

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
