"""Unit test for CTCLossStep and EarlyStopping."""

import pytest
import torch

from ocrtorch.train.loss import CTCLossStep
from ocrtorch.train.trainer import EarlyStopping


def test_ctc_loss_step_computation_and_backward():
    ctc_step = CTCLossStep(blank_id=0)

    b, t, num_classes = 2, 40, 18710
    logits = torch.randn(b, t, num_classes, requires_grad=True)
    # Targets for 2 samples
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.long)
    input_lengths = torch.tensor([40, 40], dtype=torch.long)
    target_lengths = torch.tensor([4, 2], dtype=torch.long)

    loss = ctc_step(logits, targets, input_lengths, target_lengths)
    assert loss.ndim == 0
    assert not torch.isnan(loss) and not torch.isinf(loss)

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_early_stopping_behavior():
    stopper = EarlyStopping(patience=3, min_delta=1e-3)

    # Initial score
    assert stopper.step(0.40) is True
    assert stopper.should_stop is False
    assert stopper.counter == 0

    # Improvement
    assert stopper.step(0.35) is True
    assert stopper.should_stop is False
    assert stopper.counter == 0

    # No improvement: 1st time
    assert stopper.step(0.355) is False
    assert stopper.counter == 1
    assert stopper.should_stop is False

    # No improvement: 2nd time
    assert stopper.step(0.360) is False
    assert stopper.counter == 2
    assert stopper.should_stop is False

    # No improvement: 3rd time -> Early stopping triggered!
    assert stopper.step(0.359) is False
    assert stopper.counter == 3
    assert stopper.should_stop is True


def test_trainer_saves_clean_state_dict_with_compile_and_autocast(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from ocrtorch.train.trainer import OCRTrainer

    class TinyOCR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)
            self.head = torch.nn.Linear(3, 5)

        def forward(self, x):
            # [B, 3, H, W] -> [B, 40, 5]
            pooled = self.bn(x).mean(dim=(2, 3)).unsqueeze(1).expand(-1, 40, -1)
            return self.head(pooled)

    class DictLoader:
        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    batch = {
        "images": torch.randn(2, 3, 8, 8),
        "targets": torch.tensor([[1, 2, 0, 0], [2, 1, 0, 0]], dtype=torch.long),
        "input_lengths": torch.tensor([40, 40], dtype=torch.long),
        "target_lengths": torch.tensor([2, 2], dtype=torch.long),
        "texts": ["ab", "ba"],
    }
    loader = DictLoader(batch)
    model = TinyOCR()
    trainer = OCRTrainer(
        model=model,
        train_loader=loader,  # type: ignore[arg-type]
        val_loader=loader,  # type: ignore[arg-type]
        characters=["a", "b", "c", "d"],
        device="cpu",
        output_dir=tmp_path,
        learning_rate=1e-4,
        compile_backend="eager",
        amp_dtype=torch.bfloat16,
        warmup_epochs=1,
        min_lr=1e-5,
    )
    summary = trainer.fit(epochs=2)
    assert summary["epochs_run"] == 2
    assert summary["history"][0]["lr"] == pytest.approx(1e-4)
    assert summary["history"][1]["lr"] == pytest.approx(1e-4)
    saved = torch.load(tmp_path / "latest_model.pt", weights_only=True)
    assert all(not k.startswith("_orig_mod.") for k in saved.keys())


def test_trainer_anti_forgetting_protects_unseen_ctc_and_backbone(tmp_path):
    from ocrtorch.train.trainer import OCRTrainer

    class TinyModularOCR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.bn = torch.nn.BatchNorm2d(4)
            self.ctc = torch.nn.Linear(4, 6)

        def forward(self, x):
            pooled = self.bn(self.backbone(x)).mean(dim=(2, 3)).unsqueeze(1).expand(-1, 40, -1)
            return self.ctc(pooled)

    class DictLoader:
        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    batch = {
        "images": torch.randn(2, 3, 8, 8),
        "targets": torch.tensor([[1, 2, 0, 0], [2, 1, 0, 0]], dtype=torch.long),
        "input_lengths": torch.tensor([40, 40], dtype=torch.long),
        "target_lengths": torch.tensor([2, 2], dtype=torch.long),
        "texts": ["ab", "ba"],
    }
    loader = DictLoader(batch)
    model = TinyModularOCR()
    init_backbone_w = model.backbone.weight.detach().clone()
    init_ctc_w = model.ctc.weight.detach().clone()
    init_ctc_b = model.ctc.bias.detach().clone()

    trainer = OCRTrainer(
        model=model,
        train_loader=loader,  # type: ignore[arg-type]
        val_loader=loader,  # type: ignore[arg-type]
        characters=["a", "b", "c", "d", "e"],
        device="cpu",
        output_dir=tmp_path,
        learning_rate=1e-2,
        weight_decay=0.0,
        freeze_backbone=True,
        l2_sp_weight=1e-3,
        active_token_ids={0, 1, 2},
    )
    trainer.fit(epochs=2)

    # Backbone frozen -> unchanged
    assert torch.equal(model.backbone.weight.detach(), init_backbone_w)
    # Unseen CTC rows (3, 4, 5) -> bit-for-bit unchanged
    assert torch.equal(model.ctc.weight.detach()[3:], init_ctc_w[3:])
    assert torch.equal(model.ctc.bias.detach()[3:], init_ctc_b[3:])
    # Active CTC rows (0, 1, 2) -> updated
    assert not torch.equal(model.ctc.weight.detach()[:3], init_ctc_w[:3])
