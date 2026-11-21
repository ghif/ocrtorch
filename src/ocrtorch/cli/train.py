"""Command-line interface for PP-OCRv6 fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ocrtorch.data.dataset import OCRDataset, load_character_table, ocr_collate_fn
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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv6 with PyTorch on CPU, MPS, or TPU")
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
    args = parser.parse_args()

    device = get_default_device(args.device)
    print(f"Using device: {device}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ocr_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ocr_collate_fn,
        num_workers=args.num_workers,
    )

    # Initialize model
    print("Initializing PP-OCRv6 recognizer...")
    model = PP_OCRv6MediumRec(num_classes=num_classes)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading pretrained weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Warning] Checkpoint {ckpt_path} not found, initializing from scratch.")

    trainer = OCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        characters=chars,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        patience=args.patience,
    )

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
