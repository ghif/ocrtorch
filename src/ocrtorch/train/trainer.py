"""Fine-tuning engine for PP-OCRv6 with early stopping and evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Sequence
import torch
import torch.nn as nn
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
        compile_backend: str | None = None,
        amp_dtype: torch.dtype | None = None,
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        freeze_backbone: bool = False,
        backbone_lr_scale: float = 1.0,
        l2_sp_weight: float = 0.0,
        active_token_ids: set[int] | None = None,
    ):
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        self.compile_backend = compile_backend
        # Keep master weights & AdamW states in float32; use torch.autocast for low-precision ops.
        self._raw_model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.characters = characters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.freeze_batchnorm()
        if freeze_backbone and hasattr(self._raw_model, "backbone"):
            for p in self._raw_model.backbone.parameters():
                p.requires_grad = False

        if compile_backend and compile_backend.lower() != "none":
            self.model = torch.compile(self._raw_model, backend=compile_backend)
        else:
            self.model = self._raw_model

        self.criterion = CTCLossStep().to(self.device)
        param_groups = self._build_param_groups(learning_rate, backbone_lr_scale)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.l2_sp_weight = float(l2_sp_weight)
        self._l2_sp_anchors: list[tuple[nn.Parameter, torch.Tensor]] = (
            [(p, p.detach().clone()) for p in self._raw_model.parameters() if p.requires_grad]
            if self.l2_sp_weight > 0.0
            else []
        )
        self._ctc_grad_mask: torch.Tensor | None = None
        if active_token_ids is not None and hasattr(self._raw_model, "ctc"):
            num_classes = self._raw_model.ctc.out_features
            mask = torch.zeros(num_classes, device=self.device, dtype=torch.float32)
            for idx in active_token_ids:
                if 0 <= idx < num_classes:
                    mask[idx] = 1.0
            self._ctc_grad_mask = mask

        self.early_stopping = EarlyStopping(patience=patience)
        self.max_grad_norm = max_grad_norm
        self.base_lr = learning_rate
        self.warmup_epochs = max(0, warmup_epochs)
        self.min_lr = min_lr
        self.history: list[dict] = []

    def _build_param_groups(self, learning_rate: float, backbone_lr_scale: float) -> list[dict]:
        """Build optimizer parameter groups with optional differential backbone LR."""
        if backbone_lr_scale != 1.0 and hasattr(self._raw_model, "backbone"):
            backbone_params = [p for p in self._raw_model.backbone.parameters() if p.requires_grad]
            backbone_ids = {id(p) for p in backbone_params}
            other_params = [p for p in self._raw_model.parameters() if p.requires_grad and id(p) not in backbone_ids]
            groups = []
            if backbone_params:
                groups.append({"params": backbone_params, "lr": learning_rate * backbone_lr_scale})
            if other_params:
                groups.append({"params": other_params, "lr": learning_rate})
            if groups:
                return groups
        trainable_params = [p for p in self._raw_model.parameters() if p.requires_grad]
        return [{"params": trainable_params, "lr": learning_rate}]

    def _build_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Build linear-warmup + cosine-annealing LR scheduler stepped once per epoch."""
        min_ratio = min(1.0, self.min_lr / self.base_lr) if self.base_lr > 0 else 1.0
        warmup = self.warmup_epochs

        def lr_lambda(epoch_idx: int) -> float:
            if warmup > 0 and epoch_idx < warmup:
                return float(epoch_idx + 1) / float(warmup)
            decay_epochs = max(1, epochs - warmup)
            progress = min(1.0, max(0.0, float(epoch_idx - warmup) / float(decay_epochs)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def freeze_batchnorm(self):
        """Keep BatchNorm layers frozen in eval mode with untrainable running stats.

        This preserves the fused inference checkpoint initialization from Paddle/OCRax
        and prevents gradient explosion during fine-tuning.
        """
        for m in self._raw_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.freeze_batchnorm()
        total_loss_t = torch.zeros((), device=self.device, dtype=torch.float32)
        num_batches = 0
        use_autocast = self.amp_dtype is not None

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]

            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_autocast):
                logits = self.model(images)
            loss = self.criterion(logits.float(), targets, input_lengths, target_lengths)
            if self._l2_sp_anchors:
                l2_sp_loss = sum(((p.float() - p0.float()) ** 2).sum() for p, p0 in self._l2_sp_anchors)
                loss = loss + self.l2_sp_weight * l2_sp_loss

            if self.device.type != "tpu" and (torch.isnan(loss) or torch.isinf(loss)):
                print(f"[Warning] NaN/Inf loss encountered at batch {batch_idx}, skipping update")
                continue

            loss.backward()
            if self._ctc_grad_mask is not None and hasattr(self._raw_model, "ctc"):
                ctc_layer = self._raw_model.ctc
                if ctc_layer.weight.grad is not None:
                    ctc_layer.weight.grad.mul_(self._ctc_grad_mask.unsqueeze(1))
                if ctc_layer.bias is not None and ctc_layer.bias.grad is not None:
                    ctc_layer.bias.grad.mul_(self._ctc_grad_mask)
            torch.nn.utils.clip_grad_norm_(self._raw_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss_t = total_loss_t + loss.detach().float()
            num_batches += 1

        total_loss = total_loss_t.item()
        if not math.isfinite(total_loss):
            print(f"[Warning] Non-finite training loss encountered in epoch {epoch}")
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss_t = torch.zeros((), device=self.device, dtype=torch.float32)
        num_batches = 0
        use_autocast = self.amp_dtype is not None

        all_predictions = []
        all_references = []

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]
            texts = batch["texts"]

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_autocast):
                logits = self.model(images)
            loss = self.criterion(logits.float(), targets, input_lengths, target_lengths)
            total_loss_t = total_loss_t + loss.detach().float()
            num_batches += 1

            # Greedy decode directly from argmax token IDs without materializing [B, T, V] softmax
            decoded_texts = ctc_greedy_decode(logits.argmax(dim=-1), characters=self.characters)
            all_predictions.extend(decoded_texts)
            all_references.extend(texts)

        cer = character_error_rate(all_references, all_predictions)
        raw_em, norm_em = exact_match_accuracy(all_references, all_predictions)

        return {
            "val_loss": total_loss_t.item() / max(1, num_batches),
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
        scheduler = self._build_scheduler(epochs)

        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            ep_start = time.time()
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            ep_time = time.time() - ep_start

            cer = metrics["cer"]
            improved = self.early_stopping.step(cer)

            record = {
                "epoch": epoch,
                "lr": current_lr,
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
                f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] (lr={current_lr:.2e}) - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
                f"CER: {cer*100:.2f}% | Norm EM: {metrics['norm_exact_match']*100:.2f}% {mark}"
            )
            scheduler.step()

            # Save best checkpoint
            state_to_save = self._raw_model.state_dict()
            if improved:
                best_cer = cer
                torch.save(state_to_save, self.output_dir / "best_model.pt")

            # Always save latest checkpoint
            torch.save(state_to_save, self.output_dir / "latest_model.pt")

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
