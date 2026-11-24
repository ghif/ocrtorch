"""Remote Colab bootstrap for TorchTPU fine-tuning, executed by `colab exec -f`."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATH = Path("/content/colab-finetune-config.json")
STATUS_PATH = Path("/content/colab-finetune-status.json")
CONTENT_DIR = Path("/content")
INPUTS_DIR = CONTENT_DIR / "inputs"
OUTPUT_DIR = Path("/content/fine-tuned/ppocrv6_medium_rec_rxhandbd_finetuned")

PINNED_PACKAGES = [
    "torch>=2.0.0",
    "torchvision",
    "torch-tpu",
    "numpy>=1.24.0",
    "Pillow>=9.0.0",
    "PyYAML>=6.0",
    "google-cloud-storage>=2.18,<4",
]


def run(command: list[str], *, cwd: Path | None = None, env=None) -> None:
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def capture(command: list[str]) -> str:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result.stdout


def join_parts(prefix: str, count: int, destination: Path) -> None:
    with destination.open("wb") as output:
        for index in range(count):
            part = Path(f"{prefix}{index:05d}")
            if not part.is_file():
                raise FileNotFoundError(f"Missing uploaded archive part: {part}")
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def parse_gcs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket, prefix.strip("/")


def parse_token_expiry(expiry_text: str) -> datetime:
    expiry = datetime.fromisoformat(expiry_text)
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=timezone.utc)
    return expiry.astimezone(timezone.utc)


def token_seconds_remaining(expiry_text: str) -> float:
    expiry = parse_token_expiry(expiry_text)
    return (expiry - datetime.now(timezone.utc)).total_seconds()


def download_prefix(client, uri: str, destination: Path, *, exclude_prefixes: tuple[str, ...] = ()) -> int:
    bucket_name, prefix = parse_gcs(uri)
    query = prefix.rstrip("/") + "/" if prefix else ""
    excluded = tuple(item.strip("/") + "/" for item in exclude_prefixes)
    blobs = [
        blob for blob in client.list_blobs(bucket_name, prefix=query)
        if blob.name
        and not blob.name.endswith("/")
        and not any(blob.name[len(query):].startswith(item) for item in excluded)
    ]
    if not blobs:
        raise FileNotFoundError(f"No objects found under {uri}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    download_tasks: list[tuple[object, Path]] = []
    for blob in blobs:
        target = (destination / blob.name[len(query):]).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Unsafe object path under {uri}: {blob.name}")
        download_tasks.append((blob, target))

    def _download_blob(task: tuple[object, Path]) -> None:
        blob, target = task
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_download_blob, download_tasks))
    return len(blobs)


def download_single_file(client, uri: str, destination: Path) -> None:
    bucket_name, object_name = parse_gcs(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = client.bucket(bucket_name).blob(object_name)
    blob.download_to_filename(str(destination))


def stage_gcs_inputs(client, config: dict, repo_dir: Path) -> dict:
    inputs = INPUTS_DIR
    model_dir = inputs / "model-dir"
    print(f"Reading model from {config['model_gcs']} on Colab...", flush=True)
    model_count = download_prefix(
        client,
        config["model_gcs"],
        model_dir,
        exclude_prefixes=("finetuned-runs",),
    )
    if not (model_dir / "inference.yml").is_file():
        raise FileNotFoundError(f"Expected inference.yml under {config['model_gcs']}")

    # Checkpoint staging
    checkpoint_uri = config["checkpoint_gcs"]
    checkpoint_file = inputs / "pretrained" / Path(checkpoint_uri).name
    if not checkpoint_file.is_file():
        print(f"Reading pretrained checkpoint from {checkpoint_uri}...", flush=True)
        download_single_file(client, checkpoint_uri, checkpoint_file)

    dataset_bucket, dataset_prefix = parse_gcs(config["dataset_gcs"])
    image_root = inputs / "RxHandBD" / "Train_Set"
    staged_images: dict[str, str] = {}
    split_records: dict[str, list[dict]] = {}
    download_tasks: list[tuple[str, Path]] = []

    for split in ("train", "validation"):
        source = repo_dir / "data" / "manifests" / f"{split}.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"Empty manifest: {source}")
        split_records[split] = records
        for record in records:
            filename = Path(record["image"]).name
            if not filename or filename in (".", ".."):
                raise ValueError(f"Invalid image path: {record['image']!r}")
            previous_source = staged_images.get(filename)
            if previous_source is not None and previous_source != record["image"]:
                raise ValueError(
                    f"Image basename collision while staging GCS data: {previous_source!r} "
                    f"and {record['image']!r} both map to {filename!r}"
                )
            target = image_root / filename
            if previous_source is None:
                object_name = "/".join(part for part in (dataset_prefix, "Train_Set", filename) if part)
                download_tasks.append((object_name, target))
                staged_images[filename] = record["image"]
            record["image"] = str(target)

    total_images = len(download_tasks)
    print(f"Staging {total_images} dataset images from gs://{dataset_bucket}/{dataset_prefix}...", flush=True)
    bucket = client.bucket(dataset_bucket)

    def _download_image(task: tuple[str, Path]) -> None:
        object_name, target = task
        target.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(object_name).download_to_filename(str(target))

    with ThreadPoolExecutor(max_workers=16) as pool:
        for idx, _ in enumerate(pool.map(_download_image, download_tasks), 1):
            if idx % 500 == 0 or idx == total_images:
                print(f"Staged {idx}/{total_images} images ({(idx / total_images) * 100:.1f}%)...", flush=True)

    for split, records in split_records.items():
        (repo_dir / "data" / "manifests" / f"colab_{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    return {
        "model_dir": str(model_dir),
        "checkpoint_file": str(checkpoint_file),
        "model_files": model_count,
        "images": len(staged_images),
    }


def gcs_client_from_config(config: dict):
    access_token = config.pop("gcs_access_token", None)
    if not access_token:
        raise RuntimeError("Missing short-lived GCS access token in remote config")
    expiry_text = config.pop("gcs_token_expires_at", None)
    if not expiry_text:
        raise RuntimeError("Missing GCS access token expiry in remote config")
    expiry = parse_token_expiry(expiry_text)
    if (expiry - datetime.now(timezone.utc)).total_seconds() < 900:
        raise RuntimeError("GCS access token expires in under 15 minutes; refresh local ADC and retry")

    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    client = storage.Client(
        project=config.get("gcp_project"),
        credentials=Credentials(
            token=access_token,
            expiry=expiry.replace(tzinfo=None),
            quota_project_id=config.get("quota_project_id"),
        ),
    )
    del access_token
    return client


def upload_checkpoint(gcs, config: dict) -> dict:
    output_dir = OUTPUT_DIR
    metadata_path = output_dir / "training_summary.json"
    best_model_path = output_dir / "best_model.pt"

    if not best_model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint is missing {best_model_path} or {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_bucket, output_prefix = parse_gcs(config["output_gcs"])
    uploaded = 0
    uploaded_names: set[str] = set()

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            relative_name = path.relative_to(output_dir).as_posix()
            object_name = "/".join(part for part in (output_prefix.rstrip("/"), relative_name) if part)
            gcs.bucket(output_bucket).blob(object_name).upload_from_filename(
                str(path), if_generation_match=0,
            )
            uploaded += 1
            uploaded_names.add(relative_name)

    if "best_model.pt" not in uploaded_names or "training_summary.json" not in uploaded_names:
        raise RuntimeError("Checkpoint upload did not include best_model.pt and training_summary.json")

    return {
        "status": "complete",
        "checkpoint_files": uploaded,
        "best_cer": metadata.get("best_cer"),
        "epochs_run": metadata.get("epochs_run"),
        "total_time_seconds": metadata.get("total_time_seconds"),
        "backend": metadata.get("backend", "tpu"),
        "history": metadata.get("history"),
    }


def prepare_runtime(config: dict) -> dict:
    run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        *PINNED_PACKAGES,
    ])

    # Probe TorchTPU hardware
    probe_code = (
        "import json, sys, torch; "
        "import torch_tpu; "
        "dev = torch.device('tpu'); "
        "x = torch.ones((2, 2), device=dev); "
        "res = float(x.sum().cpu()); "
        "print(json.dumps({'python': sys.version, 'backend': 'tpu', 'device': str(dev), 'res': res}))"
    )
    probe = capture([sys.executable, "-c", probe_code])
    probe_data = json.loads(probe.strip().splitlines()[-1])
    if probe_data.get("backend") != "tpu":
        raise RuntimeError(f"Expected TorchTPU 'tpu' backend, got {probe_data.get('backend')!r}")

    source_archive = Path(config["source_archive"])
    join_parts(config["source_part_prefix"], int(config["source_archive_parts"]), source_archive)
    source_root = CONTENT_DIR.resolve()

    with tarfile.open(source_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported source archive member type: {member.name}")
            target = (source_root / member.name).resolve()
            if not target.is_relative_to(source_root):
                raise ValueError(f"Unsafe source archive member: {member.name}")
        archive.extractall(source_root, filter="data")

    return {"status": "prepared", "backend": "tpu", "device": probe_data["device"]}


def validate_staged_inputs(config: dict, repo_dir: Path) -> tuple[Path, Path, Path, Path]:
    model_dir = INPUTS_DIR / "model-dir"
    if not (model_dir / "inference.yml").is_file():
        raise FileNotFoundError(f"Expected model config at {model_dir / 'inference.yml'}")

    checkpoint_uri = config["checkpoint_gcs"]
    checkpoint = INPUTS_DIR / "pretrained" / Path(checkpoint_uri).name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected staged checkpoint at {checkpoint}")

    train_manifest = repo_dir / "data" / "manifests" / "colab_train.jsonl"
    validation_manifest = repo_dir / "data" / "manifests" / "colab_validation.jsonl"
    for manifest in (train_manifest, validation_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(f"Staged manifest missing: {manifest}")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                image_path = Path(json.loads(line)["image"])
                if not image_path.is_file():
                    raise FileNotFoundError(f"Manifest image missing on Colab: {image_path}")
    return model_dir, checkpoint, train_manifest, validation_manifest


def stage_inputs(config: dict) -> dict:
    repo_dir = Path(config["source_dir"])
    if not (repo_dir / "src" / "ocrtorch" / "cli" / "train.py").is_file():
        raise FileNotFoundError("Prepared training source is missing; rerun runtime preparation")
    gcs = gcs_client_from_config(config)
    staged = stage_gcs_inputs(gcs, config, repo_dir)
    validate_staged_inputs(config, repo_dir)
    return {"status": "staged", "staged_inputs": staged}


def train_staged(config: dict) -> dict:
    repo_dir = Path(config["source_dir"])
    if not (repo_dir / "src" / "ocrtorch" / "cli" / "train.py").is_file():
        raise FileNotFoundError("Prepared training source is missing; rerun runtime preparation")

    model_dir, checkpoint, train_manifest, validation_manifest = validate_staged_inputs(config, repo_dir)
    output_dir = OUTPUT_DIR

    command = [
        sys.executable, "-m", "ocrtorch.cli.train",
        "--checkpoint", str(checkpoint),
        "--inference-yaml", str(model_dir / "inference.yml"),
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(validation_manifest),
        "--output-dir", str(output_dir),
        "--device", "tpu",
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config.get("batch_size", 64)),
        "--lr", str(config.get("lr", 4e-5)),
        "--warmup-epochs", str(config.get("warmup_epochs", 3)),
        "--min-lr", str(config.get("min_lr", 1e-6)),
        "--patience", str(config.get("patience", 8)),
        "--num-workers", str(config.get("num_workers", 4)),
        "--max-target-len", str(config.get("max_target_len", 40)),
        "--bf16" if config.get("bf16", True) else "--no-bf16",
        "--drop-last" if config.get("drop_last", True) else "--no-drop-last",
    ]

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    run(command, cwd=repo_dir, env=env)

    metadata_path = output_dir / "training_summary.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Trainer did not produce {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    summary = {
        "status": "trained",
        "checkpoint_files": sum(1 for path in output_dir.rglob("*") if path.is_file()),
        "best_cer": metadata.get("best_cer"),
        "epochs_run": metadata.get("epochs_run"),
        "total_time_seconds": metadata.get("total_time_seconds"),
        "backend": metadata.get("backend", "tpu"),
        "test_set_used": False,
        "history": metadata.get("history"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> dict:
    raw_config = CONFIG_PATH.read_text(encoding="utf-8")
    CONFIG_PATH.unlink(missing_ok=True)
    config = json.loads(raw_config)
    del raw_config

    phase = config.get("phase", "train")
    if phase == "prepare":
        return prepare_runtime(config)
    if phase == "stage":
        return stage_inputs(config)
    if phase == "upload":
        return upload_checkpoint(gcs_client_from_config(config), config)
    if phase != "train":
        raise ValueError(f"Unknown Colab run phase: {phase!r}")
    return train_staged(config)


if __name__ == "__main__":
    try:
        STATUS_PATH.unlink(missing_ok=True)
        result = main()
        STATUS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        STATUS_PATH.write_text(
            json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
