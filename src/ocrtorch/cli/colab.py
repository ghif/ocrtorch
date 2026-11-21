"""Stage and run the RxHandBD fine-tune on Colab through Google Colab CLI with TorchTPU."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ocrtorch.cloud.colab import refresh_access_token, run_colab, split_file
from ocrtorch.cloud.gcs import parse_gcs, require_gcs_prefix
from ocrtorch.data.manifests import validate_colab_manifests

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "ocrtorch"
REMOTE_DIR = "/content/handnotes-ocr"
REMOTE_ARCHIVE = "/content/handnotes-ocr.tar.gz"
REMOTE_CONFIG = "/content/colab-finetune-config.json"
REMOTE_STATUS = "/content/colab-finetune-status.json"
REMOTE_INPUT_PART = "/content/handnotes-ocr.tar.gz.part"
REMOTE_EXEC_TIMEOUT = "21600"
FINETUNED_RUNS_DIR = "finetuned-runs"

TPU_VARIANT_MAP = {
    "tpuv6e": "v6e1",
    "v6e": "v6e1",
    "v6e1": "v6e1",
    "v5e1": "v5e1",
}


def normalize_tpu_variant(variant: str) -> str:
    """Normalize user-facing TPU variant string to colab CLI argument."""
    normalized = TPU_VARIANT_MAP.get(variant.lower())
    if not normalized:
        raise ValueError(
            f"Unsupported TPU variant: {variant!r}. Choose from: {list(TPU_VARIANT_MAP.keys())}"
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-gcs", default="gs://handnotes-ocr/model-dir", help="Base GCS directory for models")
    parser.add_argument(
        "--checkpoint-gcs",
        help="Pretrained .pt checkpoint prefix; defaults to MODEL_GCS/ppocrv6_medium_rec.pt",
    )
    parser.add_argument("--dataset-gcs", default="gs://all-datasets/RxHandBD", help="GCS dataset directory")
    parser.add_argument(
        "--output-gcs",
        help="Destination override; defaults to MODEL_GCS/finetuned-runs",
    )
    parser.add_argument(
        "--tpu",
        default="tpuv6e",
        choices=["tpuv6e", "v6e", "v6e1", "v5e1"],
        help="TPU accelerator variant (default: tpuv6e)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=ROOT / "data" / "manifests",
        help="Directory containing train.jsonl, validation.jsonl, split_metadata.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    colab = shutil.which("colab")
    if not colab:
        raise SystemExit("Colab CLI is not installed or `colab` is not on PATH.")

    for uri, option in (
        (args.model_gcs, "--model-gcs"),
        (args.dataset_gcs, "--dataset-gcs"),
    ):
        try:
            require_gcs_prefix(uri, option)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    model_bucket, model_prefix = require_gcs_prefix(args.model_gcs, "--model-gcs")
    output_base = f"gs://{model_bucket}/{model_prefix}/{FINETUNED_RUNS_DIR}"
    if args.output_gcs:
        try:
            requested_output = "gs://" + "/".join(
                part for part in parse_gcs(args.output_gcs) if part
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if requested_output.rstrip("/") != output_base:
            raise SystemExit(
                f"Colab checkpoints are stored under {output_base}; "
                "omit --output-gcs or set it to that prefix."
            )

    checkpoint_gcs = args.checkpoint_gcs or f"{args.model_gcs.rstrip('/')}/ppocrv6_medium_rec.pt"
    try:
        require_gcs_prefix(checkpoint_gcs, "--checkpoint-gcs")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.epochs < 1:
        raise SystemExit("--epochs must be a positive integer")

    tpu_flag = normalize_tpu_variant(args.tpu)

    manifests_dir = args.manifests_dir
    try:
        validate_colab_manifests(manifests_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Invalid Colab training manifests: {exc}") from exc

    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise SystemExit("Install Google auth support: python -m pip install google-auth") from exc

    try:
        credentials, project = google.auth.default(scopes=[
            "openid",
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/colaboratory",
        ])
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise SystemExit(
            "Local Google ADC was not found. Run:\n"
            "  gcloud auth application-default login "
            "--scopes=openid,https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory\n"
            "on this Mac, then retry."
        ) from exc

    request = Request()
    try:
        credentials.refresh(request)
    except Exception as exc:
        raise SystemExit(
            "Could not refresh local Google ADC credentials. On this Mac, run:\n"
            "  gcloud auth application-default login "
            "--scopes=openid,https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory\n"
            f"Then retry. Details: {exc}"
        ) from exc

    run_id = datetime.now(timezone.utc).strftime("ppocrv6-rxhandbd-%Y%m%dT%H%M%S%fZ")
    session = run_id.lower()
    output_gcs = f"{output_base}/{run_id}"
    config = {
        "model_gcs": args.model_gcs,
        "checkpoint_gcs": checkpoint_gcs,
        "dataset_gcs": args.dataset_gcs,
        "output_gcs": output_gcs,
        "tpu": tpu_flag,
        "source_archive": REMOTE_ARCHIVE,
        "source_part_prefix": REMOTE_INPUT_PART,
        "source_dir": REMOTE_DIR,
        "run_id": run_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "phase": "prepare",
        "gcp_project": project,
        "quota_project_id": getattr(credentials, "quota_project_id", None),
    }

    with tempfile.TemporaryDirectory(prefix="ppocrv6-colab-") as temp_dir:
        temp = Path(temp_dir)
        archive_path = temp / "handnotes-ocr.tar.gz"
        session_config = temp / "sessions.json"
        remote_config_path = temp / "colab-finetune-config.json"

        # Bundle code and manifests
        with tarfile.open(archive_path, "w:gz") as archive:
            for source in sorted(PACKAGE.rglob("*.py")):
                archive.add(source, f"handnotes-ocr/src/ocrtorch/{source.relative_to(PACKAGE)}")
            for name in ("train.jsonl", "validation.jsonl", "split_metadata.json"):
                source = manifests_dir / name
                archive.add(source, f"handnotes-ocr/data/manifests/{name}")

        input_parts = split_file(archive_path, temp / "input-parts", "handnotes-ocr.tar.gz.part")
        config["source_archive_parts"] = len(input_parts)

        session_may_exist = False
        try:
            session_may_exist = True
            run_colab(colab, session_config, "new", "-s", session, "--tpu", tpu_flag, cwd=ROOT)
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            for index, part in enumerate(input_parts):
                run_colab(
                    colab, session_config, "upload", "-s", session, str(part),
                    f"{REMOTE_INPUT_PART}{index:05d}", cwd=ROOT,
                )
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG, cwd=ROOT)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", REMOTE_EXEC_TIMEOUT,
                cwd=ROOT,
            )
            local_status = temp / "colab-finetune-status.json"
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status), cwd=ROOT)
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "prepared":
                raise RuntimeError(f"Colab runtime preparation failed: {status.get('error', status)}")

            # Phase 2: Stage inputs
            access_token, token_expiry = refresh_access_token(credentials, request)
            config["phase"] = "stage"
            config["gcs_access_token"] = access_token
            config["gcs_token_expires_at"] = token_expiry
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG, cwd=ROOT)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", REMOTE_EXEC_TIMEOUT,
                cwd=ROOT,
            )
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status), cwd=ROOT)
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "staged":
                raise RuntimeError(f"GCS input staging did not complete: {status.get('error', status)}")

            # Phase 3: Train
            config.pop("gcs_access_token", None)
            config.pop("gcs_token_expires_at", None)
            config["phase"] = "train"
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG, cwd=ROOT)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", REMOTE_EXEC_TIMEOUT,
                cwd=ROOT,
            )
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status), cwd=ROOT)
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "trained":
                raise RuntimeError(f"Remote Colab training did not complete: {status.get('error', status)}")
            if status.get("test_set_used") is not False:
                raise RuntimeError("Remote training metadata indicates the test set was used")

            # Phase 4: Upload checkpoint
            access_token, token_expiry = refresh_access_token(credentials, request)
            config["phase"] = "upload"
            config["gcs_access_token"] = access_token
            config["gcs_token_expires_at"] = token_expiry
            remote_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_colab(colab, session_config, "upload", "-s", session, str(remote_config_path), REMOTE_CONFIG, cwd=ROOT)
            run_colab(
                colab, session_config, "exec", "-s", session,
                "-f", str(ROOT / "scripts" / "colab_remote.py"), "--timeout", "3300",
                cwd=ROOT,
            )
            run_colab(colab, session_config, "download", "-s", session, REMOTE_STATUS, str(local_status), cwd=ROOT)
            status = json.loads(local_status.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise RuntimeError(f"GCS output upload did not complete: {status.get('error', status)}")

            metadata = status
            print(json.dumps({
                "status": "complete",
                "output_gcs": output_gcs,
                "uploaded_files": metadata.get("checkpoint_files"),
                "best_epoch": metadata.get("best_epoch"),
                "best_cer": metadata.get("best_cer"),
                "backend": metadata.get("backend"),
                "gcs_inputs": {
                    "model": args.model_gcs,
                    "checkpoint": checkpoint_gcs,
                    "dataset": args.dataset_gcs,
                },
            }, indent=2), flush=True)

        finally:
            if session_may_exist:
                try:
                    run_colab(colab, session_config, "stop", "-s", session, cwd=ROOT)
                except Exception as exc:
                    print(f"Warning: failed to stop Colab session {session}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
