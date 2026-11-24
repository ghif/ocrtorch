"""Tests for Buku KIA page 3 manifest preparation and checkpoint interpolation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import torch

from scripts.merge_checkpoints import interpolate_state_dicts
from scripts.prepare_bukukia_page3 import prepare_manifests


def _write_crop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 24), color=(255, 255, 255)).save(path)


def test_prepare_manifests_task_grouped_no_leakage(tmp_path):
    ds_dir = tmp_path / "dataset_fine_tunning"
    ds_dir.mkdir()

    # 10 train tasks across 2 templates + 2 test tasks
    assignments = []
    for i in range(1, 11):
        assignments.append({
            "task_id": str(1000 + i),
            "split": "train_set",
            "template_family": "bukukia2023_page1" if i <= 5 else "bukukia2024_page96",
            "has_hemoglobin": "True" if i % 3 == 0 else "False",
        })
    for i in range(11, 13):
        assignments.append({
            "task_id": str(1000 + i),
            "split": "test_set",
            "template_family": "bukukia2023_page1",
            "has_hemoglobin": "False",
        })

    with (ds_dir / "split_assignments.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "split", "template_family", "has_hemoglobin"])
        writer.writeheader()
        writer.writerows(assignments)

    for split_name, tids in (("train_set", range(1001, 1011)), ("test_set", range(1011, 1013))):
        for field in ("dates", "blood_pressure", "hemoglobin"):
            rows = []
            for tid in tids:
                if field == "hemoglobin" and tid % 3 != 0 and split_name == "train_set":
                    continue
                fname = f"task{tid}_box000.png"
                _write_crop(ds_dir / split_name / field / fname)
                rows.append({"image_path": fname, "value": "12/05/25" if field == "dates" else "110/70"})
            with (ds_dir / split_name / field / "labels.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "value"])
                writer.writeheader()
                writer.writerows(rows)

    out_dir = tmp_path / "manifests"
    meta = prepare_manifests(ds_dir, out_dir, val_fraction=0.2, seed=140)
    assert meta["train"]["unique_tasks"] + meta["val"]["unique_tasks"] == 10
    assert meta["test"]["unique_tasks"] == 2

    train_tids = {json.loads(line)["task_id"] for line in (out_dir / "train.jsonl").read_text().splitlines()}
    val_tids = {json.loads(line)["task_id"] for line in (out_dir / "val.jsonl").read_text().splitlines()}
    test_tids = {json.loads(line)["task_id"] for line in (out_dir / "test.jsonl").read_text().splitlines()}
    assert not (train_tids & val_tids)
    assert not (train_tids & test_tids)
    assert not (val_tids & test_tids)


def test_interpolate_state_dicts():
    base = {"w": torch.tensor([0.0, 2.0]), "num_batches_tracked": torch.tensor(10)}
    ft = {"w": torch.tensor([2.0, 6.0]), "num_batches_tracked": torch.tensor(20)}
    merged = interpolate_state_dicts(base, ft, alpha=0.25)
    assert torch.allclose(merged["w"], torch.tensor([0.5, 3.0]))
    assert torch.equal(merged["num_batches_tracked"], torch.tensor(20))
