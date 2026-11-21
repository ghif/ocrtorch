"""Unit tests for remote Colab runner in ocrtorch."""

from __future__ import annotations

import io
import json
import tempfile
import tarfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import colab_remote
from ocrtorch.cloud.gcs import require_gcs_prefix, parse_gcs


class FakeBlob:
    def __init__(self, client: "FakeGCS", object_key: str, name: str):
        self.client = client
        self.object_key = object_key
        self.name = name

    def download_to_filename(self, filename: str) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.client.objects[self.object_key])

    def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
        if if_generation_match != 0:
            raise AssertionError("GCS outputs must use create-only preconditions")
        if self.name in self.client.objects:
            raise FileExistsError(self.name)
        self.client.objects[self.name] = Path(filename).read_bytes()


class FakeBucket:
    def __init__(self, client: "FakeGCS", name: str):
        self.client = client
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.client, f"{self.name}/{name}", name)


class FakeGCS:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = objects or {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)

    def list_blobs(self, bucket: str, *, prefix: str):
        return [
            FakeBlob(self, key, key[len(f"{bucket}/"):])
            for key in sorted(self.objects)
            if key.startswith(f"{bucket}/{prefix}")
        ]


class ColabRemoteTests(unittest.TestCase):
    def test_token_expiry_accepts_naive_and_aware_utc_values(self) -> None:
        for expiry in (
            (datetime.now(timezone.utc) + timedelta(minutes=20)).replace(tzinfo=None),
            datetime.now(timezone.utc) + timedelta(minutes=20),
        ):
            with self.subTest(expiry=expiry):
                remaining = colab_remote.token_seconds_remaining(expiry.isoformat())
                self.assertGreater(remaining, 900)

    def test_parse_gcs_requires_bucket_name(self) -> None:
        self.assertEqual(parse_gcs("gs://bucket/a/b/"), ("bucket", "a/b"))
        for uri in ("https://bucket/path", "gs:///path", "gs://"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                parse_gcs(uri)

    def test_model_and_dataset_uris_require_a_prefix(self) -> None:
        self.assertEqual(require_gcs_prefix("gs://bucket/path", "--model-gcs"), ("bucket", "path"))
        for uri in ("gs://bucket", "gs://bucket/"):
            with self.subTest(uri=uri), self.assertRaisesRegex(ValueError, "object prefix"):
                require_gcs_prefix(uri, "--model-gcs")

    def test_remote_gcs_credentials_require_token_expiry(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Missing GCS access token expiry"):
            colab_remote.gcs_client_from_config({"gcs_access_token": "short-lived-token"})

    def test_gcs_client_tracks_token_expiry_and_consumes_secrets_from_config(self) -> None:
        naive_expiry = (datetime.now(timezone.utc) + timedelta(minutes=30)).replace(tzinfo=None)
        expiry = naive_expiry.replace(tzinfo=timezone.utc)
        config = {
            "gcs_access_token": "short-lived-token",
            "gcs_token_expires_at": naive_expiry.isoformat(),
            "gcp_project": "quota-project",
            "quota_project_id": "quota-project",
        }
        client = colab_remote.gcs_client_from_config(config)
        self.assertEqual(client._credentials.token, "short-lived-token")
        actual_expiry = client._credentials.expiry
        if actual_expiry.tzinfo is None:
            actual_expiry = actual_expiry.replace(tzinfo=timezone.utc)
        self.assertEqual(actual_expiry, expiry)
        self.assertEqual(client._credentials.quota_project_id, "quota-project")
        self.assertNotIn("gcs_access_token", config)
        self.assertNotIn("gcs_token_expires_at", config)

    def test_download_prefix_rejects_path_traversal(self) -> None:
        client = FakeGCS({"bucket/prefix/../../escape": b"bad"})
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Unsafe object path"):
                colab_remote.download_prefix(client, "gs://bucket/prefix", Path(temp) / "out")

    def test_stage_inputs_downloads_from_gcs_and_deduplicates_images(self) -> None:
        objects = {
            "models/base/inference.yml": b"model config",
            "models/base/ppocrv6_medium_rec.pt": b"pt weights",
            "datasets/RxHandBD/Train_Set/P1.jpg": b"image bytes",
        }
        client = FakeGCS(objects)
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            previous_inputs = colab_remote.INPUTS_DIR
            colab_remote.INPUTS_DIR = Path(temp) / "inputs"
            manifests = repo / "data" / "manifests"
            manifests.mkdir(parents=True)
            row = {"image": "/host/path/P1.jpg", "text": "label"}
            for split in ("train", "validation"):
                (manifests / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            try:
                staged = colab_remote.stage_gcs_inputs(client, {
                    "model_gcs": "gs://models/base",
                    "checkpoint_gcs": "gs://models/base/ppocrv6_medium_rec.pt",
                    "dataset_gcs": "gs://datasets/RxHandBD",
                }, repo)

                self.assertEqual(staged["model_files"], 2)
                self.assertEqual(staged["images"], 1)
                staged_image = colab_remote.INPUTS_DIR / "RxHandBD" / "Train_Set" / "P1.jpg"
                train_rows = [json.loads(line) for line in (manifests / "colab_train.jsonl").read_text().splitlines()]
                validation_rows = [json.loads(line) for line in (manifests / "colab_validation.jsonl").read_text().splitlines()]
                self.assertEqual(train_rows[0]["image"], str(staged_image))
                self.assertEqual(validation_rows[0]["image"], str(staged_image))
                self.assertEqual(staged_image.read_bytes(), b"image bytes")
            finally:
                colab_remote.INPUTS_DIR = previous_inputs

    @patch("scripts.colab_remote.subprocess.run")
    @patch("scripts.colab_remote.capture", return_value='{"backend":"tpu","device":"tpu:0"}')
    @patch("scripts.colab_remote.run")
    def test_prepare_runtime_verifies_tpu_and_extracts_source(self, run_command, _capture, _run_subprocess) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_archive = root / "source.tar.gz"
            with tarfile.open(source_archive, "w:gz") as archive:
                payload = b"# staged source\n"
                info = tarfile.TarInfo("handnotes-ocr/src/ocrtorch/cli/train.py")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            part_prefix = root / "source.part"
            (root / "source.part00000").write_bytes(source_archive.read_bytes())
            assembled_archive = root / "assembled.tar.gz"
            previous_content_dir = colab_remote.CONTENT_DIR
            colab_remote.CONTENT_DIR = root / "content"
            try:
                result = colab_remote.prepare_runtime({
                    "source_archive": str(assembled_archive),
                    "source_part_prefix": str(part_prefix),
                    "source_archive_parts": 1,
                })
                self.assertEqual(result, {"status": "prepared", "backend": "tpu", "device": "tpu:0"})
                staged_source = root / "content/handnotes-ocr/src/ocrtorch/cli/train.py"
                self.assertEqual(staged_source.read_bytes(), b"# staged source\n")
                run_command.assert_called_once()
            finally:
                colab_remote.CONTENT_DIR = previous_content_dir

    @patch("scripts.colab_remote.run")
    def test_training_executes_on_tpu(self, run_training) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            package = repo / "src" / "ocrtorch"
            manifests = repo / "data" / "manifests"
            manifests.mkdir(parents=True)
            (package / "cli" / "train.py").parent.mkdir(parents=True, exist_ok=True)
            (package / "cli" / "train.py").write_text("# fake trainer\n", encoding="utf-8")
            image = root / "P1.jpg"
            image.write_bytes(b"image")
            row = json.dumps({"image": str(image), "text": "label"}) + "\n"
            for split in ("train", "validation"):
                (manifests / f"colab_{split}.jsonl").write_text(row, encoding="utf-8")

            inputs = root / "inputs"
            model_dir = inputs / "model-dir"
            model_dir.mkdir(parents=True)
            (model_dir / "inference.yml").write_text("model", encoding="utf-8")
            ckpt_dir = inputs / "pretrained"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "ppocrv6_medium_rec.pt").write_bytes(b"weights")
            output = root / "output"
            previous_inputs, previous_output = colab_remote.INPUTS_DIR, colab_remote.OUTPUT_DIR
            colab_remote.INPUTS_DIR, colab_remote.OUTPUT_DIR = inputs, output

            def create_trainer_output(*_args, **_kwargs):
                output.mkdir(parents=True)
                (output / "best_model.pt").write_bytes(b"weights")
                (output / "training_summary.json").write_text(json.dumps({
                    "best_cer": 0.05,
                    "epochs_run": 1,
                    "backend": "tpu",
                }), encoding="utf-8")

            run_training.side_effect = create_trainer_output
            try:
                result = colab_remote.train_staged({
                    "source_dir": str(repo),
                    "model_gcs": "gs://models/base",
                    "checkpoint_gcs": "gs://models/base/ppocrv6_medium_rec.pt",
                    "epochs": 1,
                })
                self.assertEqual(result["status"], "trained")
                self.assertEqual(result["backend"], "tpu")
                self.assertEqual(run_training.call_count, 1)
                cmd = run_training.call_args.args[0]
                self.assertIn("--device", cmd)
                self.assertIn("tpu", cmd)
            finally:
                colab_remote.INPUTS_DIR, colab_remote.OUTPUT_DIR = previous_inputs, previous_output

    def test_upload_saves_pytorch_checkpoints(self) -> None:
        class NoListGCS(FakeGCS):
            def list_blobs(self, *_args, **_kwargs):
                raise AssertionError("output bucket listing must not be required")

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "best_model.pt").write_bytes(b"best weights")
            (output / "latest_model.pt").write_bytes(b"latest weights")
            (output / "training_summary.json").write_text(json.dumps({
                "best_cer": 0.03,
                "epochs_run": 5,
                "backend": "tpu",
            }), encoding="utf-8")
            previous = colab_remote.OUTPUT_DIR
            colab_remote.OUTPUT_DIR = output
            try:
                client = NoListGCS()
                result = colab_remote.upload_checkpoint(client, {"output_gcs": "gs://output/run-1"})
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["checkpoint_files"], 3)
                self.assertEqual(set(client.objects), {
                    "run-1/best_model.pt",
                    "run-1/latest_model.pt",
                    "run-1/training_summary.json",
                })
            finally:
                colab_remote.OUTPUT_DIR = previous


if __name__ == "__main__":
    unittest.main()
