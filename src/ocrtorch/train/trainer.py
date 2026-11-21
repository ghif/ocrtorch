"""Fine-tuning engine for PP-OCRv6 with early stopping and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ocrtorch.data.vocab import ctc_greedy_decode
from ocrtorch.domain.metrics import character_error_rate, exact_match_accuracy
from ocrtorch.train.loss import CTCLossStep


class EarlyStopping:
    """Early stopping tracker monitoring validation metric (default: CER, lower is better)."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, current_score: float) -> bool:
        """Update tracker with new validation score. Returns True if improved."""
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
            return True

        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class OCRTrainer:
    """Trainer handling training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        characters: Sequence[str],
        device: torch.device | str,
        output_dir: Path | str,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-5,
        patience: int = 3,
        max_grad_norm: float = 1.0,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.characters = characters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.freeze_batchnorm()

        self.criterion = CTCLossStep().to(self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.early_stopping = EarlyStopping(patience=patience)
        self.max_grad_norm = max_grad_norm
        self.history: list[dict] = []

    def freeze_batchnorm(self):
        """Keep BatchNorm layers frozen in eval mode with untrainable running stats.

        This preserves the fused inference checkpoint initialization from Paddle/OCRax
        and prevents gradient explosion during fine-tuning.
        """
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.freeze_batchnorm()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss encountered at batch {batch_idx}, skipping update")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_references = []

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]
            texts = batch["texts"]

            logits = self.model(images)
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1

            # Decode greedy predictions using softmax probabilities
            probs = F.softmax(logits.float(), dim=-1)
            decoded_texts = ctc_greedy_decode(probs, characters=self.characters)
            all_predictions.extend(decoded_texts)
            all_references.extend(texts)

        cer = character_error_rate(all_references, all_predictions)
        raw_em, norm_em = exact_match_accuracy(all_references, all_predictions)

        return {
            "val_loss": total_loss / max(1, num_batches),
            "cer": cer,
            "raw_exact_match": raw_em,
            "norm_exact_match": norm_em,
        }

    def fit(self, epochs: int) -> dict:
        print(f"Starting training on device: {self.device} for {epochs} epochs (Early stopping patience: {self.early_stopping.patience})")
        start_time = time.time()
        best_cer = float("inf")

        # Initial zero-shot evaluation
        init_metrics = self.evaluate()
        print(
            f"Zero-shot Pretrained Baseline - Val Loss: {init_metrics['val_loss']:.4f} | "
            f"CER: {init_metrics['cer']*100:.2f}% | Norm EM: {init_metrics['norm_exact_match']*100:.2f}%"
        )
        self.early_stopping.step(init_metrics["cer"])
        best_cer = init_metrics["cer"]

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            ep_time = time.time() - ep_start

            cer = metrics["cer"]
            improved = self.early_stopping.step(cer)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["val_loss"],
                "cer": cer,
                "norm_exact_match": metrics["norm_exact_match"],
                "time_sec": ep_time,
                "improved": improved,
            }
            self.history.append(record)

            mark = "★ (Best)" if improved else f"(Patience {self.early_stopping.counter}/{self.early_stopping.patience})"
            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
                f"CER: {cer*100:.2f}% | Norm EM: {metrics['norm_exact_match']*100:.2f}% {mark}"
            )

            # Save best checkpoint
            if improved:
                best_cer = cer
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")

            # Always save latest checkpoint
            torch.save(self.model.state_dict(), self.output_dir / "latest_model.pt")

            # Check early stopping
            if self.early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch}: Validation CER did not improve for {self.early_stopping.patience} epochs.")
                break

        total_time = time.time() - start_time
        summary = {
            "backend": self.device.type,
            "device": str(self.device),
            "total_time_seconds": total_time,
            "epochs_run": len(self.history),
            "best_cer": best_cer,
            "early_stopped": self.early_stopping.should_stop,
            "history": self.history,
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Training finished in {total_time:.1f}s. Best Validation CER: {best_cer*100:.2f}%. Saved to {self.output_dir}")
        return summary
