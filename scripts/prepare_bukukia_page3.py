"""Prepare task-grouped train/val/test JSONL manifests for Buku KIA page 3."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import re

FIELDS = ("dates", "blood_pressure", "hemoglobin")
TASK_ID_RE = re.compile(r"^task(\d+)_")


def extract_task_id(image_filename: str) -> str:
    """Extract numeric task_id from crop filename like 'task33954_box000.png'."""
    match = TASK_ID_RE.match(Path(image_filename).name)
    if not match:
        raise ValueError(f"Could not extract task_id from image filename: {image_filename!r}")
    return match.group(1)


def load_split_assignments(csv_path: Path) -> dict[str, dict[str, str]]:
    """Load task_id -> {split, template_family, has_hemoglobin} mapping."""
    assignments: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            task_id = row["task_id"].strip()
            assignments[task_id] = {
                "split": row["split"].strip(),
                "template_family": row.get("template_family", "").strip(),
                "has_hemoglobin": row.get("has_hemoglobin", "").strip(),
            }
    return assignments


def collect_split_rows(
    dataset_dir: Path,
    split_folder: str,
    assignments: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Collect {image, text, field, task_id, template_family} rows for train_set or test_set."""
    rows: list[dict[str, str]] = []
    for field in FIELDS:
        field_dir = dataset_dir / split_folder / field
        labels_path = field_dir / "labels.csv"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        with labels_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rel_img = row["image_path"].strip()
                text = row["value"]
                img_path = (field_dir / rel_img).resolve()
                if not img_path.is_file():
                    raise FileNotFoundError(f"Missing crop image: {img_path}")
                task_id = extract_task_id(rel_img)
                meta = assignments.get(task_id, {})
                rows.append(
                    {
                        "image": str(img_path),
                        "text": text,
                        "field": field,
                        "task_id": task_id,
                        "template_family": meta.get("template_family", ""),
                    }
                )
    return rows


def stratified_task_split(
    train_rows: list[dict[str, str]],
    assignments: dict[str, dict[str, str]],
    val_fraction: float = 0.15,
    seed: int = 140,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split train_rows into train/val grouped by task_id and stratified by (template_family, has_hemoglobin)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    strata_to_tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in train_rows:
        tid = row["task_id"]
        meta = assignments.get(tid, {})
        key = (meta.get("template_family", ""), meta.get("has_hemoglobin", ""))
        strata_to_tasks[key].add(tid)

    rng = random.Random(seed)
    val_tasks: set[str] = set()
    for key in sorted(strata_to_tasks):
        tasks = sorted(strata_to_tasks[key])
        rng.shuffle(tasks)
        n_val = max(1, int(round(len(tasks) * val_fraction))) if len(tasks) > 1 else 0
        val_tasks.update(tasks[:n_val])

    out_train = [r for r in train_rows if r["task_id"] not in val_tasks]
    out_val = [r for r in train_rows if r["task_id"] in val_tasks]
    return out_train, out_val


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict[str, str]]) -> dict:
    """Compute per-field and unique-task counts."""
    by_field: dict[str, int] = defaultdict(int)
    tasks: set[str] = set()
    for r in rows:
        by_field[r["field"]] += 1
        tasks.add(r["task_id"])
    return {
        "total": len(rows),
        "unique_tasks": len(tasks),
        "by_field": dict(sorted(by_field.items())),
    }


def prepare_manifests(
    dataset_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 140,
) -> dict:
    """Build train.jsonl, val.jsonl, test.jsonl, and split_metadata.json."""
    assignments = load_split_assignments(dataset_dir / "split_assignments.csv")
    raw_train_rows = collect_split_rows(dataset_dir, "train_set", assignments)
    test_rows = collect_split_rows(dataset_dir, "test_set", assignments)

    train_rows, val_rows = stratified_task_split(
        raw_train_rows,
        assignments=assignments,
        val_fraction=val_fraction,
        seed=seed,
    )

    # Verify zero task_id overlap across all three splits
    train_tasks = {r["task_id"] for r in train_rows}
    val_tasks = {r["task_id"] for r in val_rows}
    test_tasks = {r["task_id"] for r in test_rows}
    assert not (train_tasks & val_tasks), "Task leakage between train and val!"
    assert not (train_tasks & test_tasks), "Task leakage between train and test!"
    assert not (val_tasks & test_tasks), "Task leakage between val and test!"

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    metadata = {
        "dataset": "bukukia_page3",
        "source_dir": str(dataset_dir.resolve()),
        "seed": seed,
        "val_fraction": val_fraction,
        "split_unit": "task_id",
        "train": summarize_rows(train_rows),
        "val": summarize_rows(val_rows),
        "test": summarize_rows(test_rows),
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Buku KIA page 3 JSONL manifests")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Local directory containing staged dataset_fine_tunning/ (train_set, test_set, split_assignments.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests/bukukia_page3"),
        help="Output directory for train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of train_set tasks for validation")
    parser.add_argument("--seed", type=int, default=140, help="Random seed for task-grouped val split")
    args = parser.parse_args()

    metadata = prepare_manifests(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
