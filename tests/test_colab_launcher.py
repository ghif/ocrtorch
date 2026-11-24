"""Unit tests for ocrtorch Colab CLI launcher."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import google.auth

from ocrtorch.cli import colab as colab_launcher
from ocrtorch.cli.colab import normalize_tpu_variant


class FakeCredentials:
    quota_project_id = "quota-project"
    token = None
    expiry = None
    refresh_count = 0

    def refresh(self, _request) -> None:
        self.refresh_count += 1
        self.token = f"fake-access-token-{self.refresh_count}"
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


class ColabLauncherTests(unittest.TestCase):
    def test_normalize_tpu_variant(self) -> None:
        self.assertEqual(normalize_tpu_variant("tpuv6e"), "v6e1")
        self.assertEqual(normalize_tpu_variant("v6e"), "v6e1")
        self.assertEqual(normalize_tpu_variant("v6e1"), "v6e1")
        self.assertEqual(normalize_tpu_variant("tpuv5e"), "v5e1")
        self.assertEqual(normalize_tpu_variant("v5e"), "v5e1")
        self.assertEqual(normalize_tpu_variant("v5e1"), "v5e1")
        with self.assertRaises(ValueError):
            normalize_tpu_variant("unsupported_tpu")

    def test_parse_args_loads_yaml_config_and_allows_cli_override(self) -> None:
        cfg_path = colab_launcher.ROOT / "configs" / "tpu_v6e.yaml"
        args = colab_launcher.parse_args([
            "--config", str(cfg_path),
            "--tpu", "v5e",
            "--epochs", "10",
        ])
        self.assertEqual(args.tpu, "v5e")
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.lr, 4e-5)
        self.assertEqual(args.warmup_epochs, 3)
        self.assertEqual(args.min_lr, 1e-6)
        self.assertEqual(args.patience, 8)
        self.assertEqual(args.num_workers, 4)
        self.assertTrue(args.bf16)
        self.assertEqual(args.max_target_len, 40)
        self.assertTrue(args.drop_last)

    def test_main_runs_phases_with_tpu_and_stops_session(self) -> None:
        phases: list[str] = []
        commands: list[tuple[str, ...]] = []
        uploaded_configs: dict[str, dict] = {}
        uploaded_archive_parts: list[tuple[int, bytes]] = []
        statuses: dict[str, dict] = {
            "prepare": {"status": "prepared", "backend": "tpu", "device": "tpu:0"},
            "stage": {"status": "staged", "staged_inputs": {"images": 2}},
            "train": {"status": "trained", "test_set_used": False, "best_cer": 0.05},
            "upload": {"status": "complete", "checkpoint_files": 3, "best_cer": 0.05},
        }
        active_phase = "prepare"
        stopped = False

        def fake_run_colab(_colab: str, _session_config: Path, *args: str, cwd: Path | None = None) -> None:
            nonlocal active_phase, stopped
            commands.append(tuple(args))
            operation = args[0]
            local_upload = Path(args[-2]) if operation == "upload" else None
            if local_upload and local_upload.name.startswith("handnotes-ocr.tar.gz.part"):
                part_index = int(local_upload.name.removeprefix("handnotes-ocr.tar.gz.part"))
                uploaded_archive_parts.append((part_index, local_upload.read_bytes()))
            if operation == "upload" and Path(args[-2]).name == "colab-finetune-config.json":
                uploaded_config = json.loads(Path(args[-2]).read_text(encoding="utf-8"))
                active_phase = uploaded_config["phase"]
                uploaded_configs[active_phase] = uploaded_config
            elif operation == "exec":
                phases.append(active_phase)
            elif operation == "download":
                Path(args[-1]).write_text(json.dumps(statuses[active_phase]), encoding="utf-8")
            elif operation == "stop":
                stopped = True

        credentials = FakeCredentials()
        with (
            patch.object(colab_launcher.shutil, "which", return_value="/usr/bin/colab"),
            patch.object(colab_launcher, "run_colab", side_effect=fake_run_colab),
            patch.object(google.auth, "default", return_value=(credentials, "quota-project")),
            patch.object(sys, "argv", [
                "colab_launcher",
                "--model-gcs", "gs://models/base",
                "--dataset-gcs", "gs://datasets/RxHandBD",
                "--output-gcs", "gs://models/base/finetuned-runs",
                "--tpu", "tpuv6e",
                "--epochs", "2",
            ]),
        ):
            self.assertEqual(colab_launcher.main(), 0)

        self.assertEqual(phases, ["prepare", "stage", "train", "upload"])
        self.assertEqual(credentials.refresh_count, 3)
        self.assertIn("gcs_access_token", uploaded_configs["stage"])
        self.assertNotIn("gcs_access_token", uploaded_configs["train"])
        self.assertIn("gcs_access_token", uploaded_configs["upload"])
        self.assertTrue(stopped)

        # Verify colab new was called with --tpu v6e1
        new_command = next(cmd for cmd in commands if cmd[0] == "new")
        self.assertIn("--tpu", new_command)
        self.assertIn("v6e1", new_command)

        # Verify archive contains ocrtorch sources
        archive_bytes = b"".join(part for _, part in sorted(uploaded_archive_parts))
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            archived_names = set(archive.getnames())
        expected_sample_source = f"handnotes-ocr/src/ocrtorch/cli/train.py"
        self.assertIn(expected_sample_source, archived_names)

    def test_main_attempts_cleanup_if_session_creation_errors(self) -> None:
        operations: list[str] = []

        def fake_run_colab(_colab: str, _session_config: Path, *args: str, cwd: Path | None = None) -> None:
            operations.append(args[0])
            if args[0] == "new":
                raise subprocess.CalledProcessError(1, ["colab", *args])

        with (
            patch.object(colab_launcher.shutil, "which", return_value="/usr/bin/colab"),
            patch.object(colab_launcher, "run_colab", side_effect=fake_run_colab),
            patch.object(google.auth, "default", return_value=(FakeCredentials(), "quota-project")),
            patch.object(sys, "argv", [
                "colab_launcher",
                "--model-gcs", "gs://models/base",
                "--dataset-gcs", "gs://datasets/RxHandBD",
            ]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                colab_launcher.main()

        self.assertEqual(operations, ["new", "stop"])


if __name__ == "__main__":
    unittest.main()
