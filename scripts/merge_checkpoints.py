"""Weight-space interpolation (Model Soup) and per-field evaluation for PP-OCRv6."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ocrtorch.cli.evaluate import EvalOCRDataset, eval_collate_fn
from ocrtorch.cli.train import get_default_device
from ocrtorch.data.dataset import load_character_table, load_manifest
from ocrtorch.data.vocab import ctc_greedy_decode
from ocrtorch.domain.metrics import character_error_rate, exact_match_accuracy
from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec


def interpolate_state_dicts(
    base_state: dict[str, torch.Tensor],
    ft_state: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Compute (1 - alpha) * base + alpha * ft for floating-point tensors."""
    merged: dict[str, torch.Tensor] = {}
    for key, base_tensor in base_state.items():
        if key not in ft_state:
            merged[key] = base_tensor
            continue
        ft_tensor = ft_state[key]
        if base_tensor.is_floating_point():
            merged[key] = ((1.0 - alpha) * base_tensor.float() + alpha * ft_tensor.float()).to(base_tensor.dtype)
        else:
            merged[key] = ft_tensor
    return merged


@torch.no_grad()
def evaluate_by_slice(
    model: PP_OCRv6MediumRec,
    rows: list[dict[str, str]],
    chars: list[str],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Evaluate model overall and broken down by 'field' and 'template_family'."""
    model.eval()
    loader = DataLoader(
        EvalOCRDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate_fn,
    )
    preds: list[str] = []
    refs: list[str] = []
    for batch in loader:
        logits = model(batch["images"].to(device))
        preds.extend(ctc_greedy_decode(logits.argmax(dim=-1), characters=chars))
        refs.extend(batch["texts"])

    def _metrics(indices: list[int]) -> dict[str, float | int]:
        sub_refs = [refs[i] for i in indices]
        sub_preds = [preds[i] for i in indices]
        raw_em, norm_em = exact_match_accuracy(sub_refs, sub_preds)
        return {
            "count": len(indices),
            "cer": character_error_rate(sub_refs, sub_preds),
            "raw_exact_match": raw_em,
            "norm_exact_match": norm_em,
        }

    result: dict = {"overall": _metrics(list(range(len(rows))))}
    for slice_key in ("field", "template_family"):
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            val = row.get(slice_key)
            if val:
                groups[val].append(idx)
        if groups:
            result[f"by_{slice_key}"] = {k: _metrics(idxs) for k, idxs in sorted(groups.items())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate base & fine-tuned checkpoints and evaluate by field")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="Pretrained base .pt checkpoint")
    parser.add_argument("--ft-checkpoint", type=Path, required=True, help="Fine-tuned .pt checkpoint")
    parser.add_argument("--inference-yaml", type=Path, required=True, help="Path to Paddle inference.yml")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to test.jsonl or val.jsonl")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 0.7, 1.0], help="Interpolation weights")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "tpu", "mps", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional dir to save merged checkpoints and report")
    args = parser.parse_args()

    device = get_default_device(args.device)
    chars, _ = load_character_table(args.inference_yaml)
    num_classes = len(chars) + 1
    rows = load_manifest(args.manifest)

    base_state = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    ft_state = torch.load(args.ft_checkpoint, map_location="cpu", weights_only=True)

    report: dict[str, dict] = {}
    for alpha in args.alphas:
        merged_state = interpolate_state_dicts(base_state, ft_state, alpha=alpha)
        model = PP_OCRv6MediumRec(num_classes=num_classes)
        model.load_state_dict(merged_state, strict=False)
        model = model.to(device)
        metrics = evaluate_by_slice(
            model,
            rows=rows,
            chars=chars,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        key = f"alpha_{alpha:.2f}"
        report[key] = metrics
        overall = metrics["overall"]
        print(
            f"[{key}] CER={overall['cer']*100:.2f}% | "
            f"RawEM={overall['raw_exact_match']*100:.2f}% | "
            f"NormEM={overall['norm_exact_match']*100:.2f}%"
        )
        if args.output_dir is not None and 0.0 < alpha < 1.0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(merged_state, args.output_dir / f"merged_{key}.pt")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "merge_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
