from __future__ import annotations

from pathlib import Path
import os
import sys
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eventlog import event, log_context, stage
from media import run
from runtime import gpu_fields
from common import PROJECT_ROOT
from download_models import download_models


class LocalJournalTests(unittest.TestCase):
    def test_model_download_reports_stages_and_preserves_the_revision_lock(self):
        root = PROJECT_ROOT / ".test-tmp"
        root.mkdir(exist_ok=True)
        entries = []
        model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        # Duplicate FP32 shards would exceed any disk if they were counted or downloaded.
        info = SimpleNamespace(sha="abc123", siblings=[
            SimpleNamespace(size=100, rfilename="model.safetensors"),
            SimpleNamespace(size=10 ** 18, rfilename="model.fp32-00001-of-00002.safetensors")])

        def snapshot(**kwargs):
            self.assertIn("*fp32*", kwargs["ignore_patterns"])
            directory = Path(kwargs["local_dir"])
            directory.mkdir(parents=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"placeholder; no real model downloaded")

        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info), snapshot_download=snapshot)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            config = {"modelsDir": temp, "asrModel": model_id, "asrAlignerModel": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
                      "visionModel": "Qwen/Qwen3.5-9B"}
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(os.environ), log_context(sink=entries.append):
                lock = download_models({"config": config, "roles": ["asr"]})
            self.assertEqual(lock["models"]["asr"]["revision"], "abc123")
            self.assertFalse(Path(lock["models"]["asr"]["path"]).is_absolute())
            names = [entry["name"] for entry in entries]
            self.assertIn("models.resolve.start", names)
            self.assertIn("models.download.plan", names)
            self.assertIn("models.download.complete", names)
            self.assertTrue(all(entry["fields"].get("modelId") == model_id for entry in entries))

    def test_heartbeat_keeps_context_and_stops_after_completion(self):
        entries = []
        waiting = threading.Event()

        def sink(entry):
            entries.append(entry)
            if entry["name"].endswith(".waiting"):
                waiting.set()

        with log_context(sink=sink, videoId="abc123def45", chunk=3):
            with stage("vision.generate", "Analyse", heartbeat_seconds=0.01):
                self.assertTrue(waiting.wait(1))
        count = len(entries)
        time.sleep(0.03)
        self.assertEqual(len(entries), count)
        self.assertEqual(entries[0]["name"], "vision.generate.start")
        self.assertEqual(entries[-1]["name"], "vision.generate.complete")
        self.assertTrue(all(entry["fields"]["chunk"] == 3 for entry in entries))
        self.assertIn("elapsedMs", next(entry["fields"] for entry in entries if entry["name"].endswith(".waiting")))

    def test_error_preserves_exception_and_redacts_tokens(self):
        entries = []
        error = ValueError("clé hf_abcdefghijklmnopqrstuvwxyz")
        with log_context(sink=entries.append):
            with self.assertRaises(ValueError) as caught:
                with stage("audio.generate", "Transcription"):
                    raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(entries[-1]["level"], "error")
        self.assertEqual(entries[-1]["fields"]["errorType"], "ValueError")
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz", entries[-1]["message"])

    def test_media_subprocess_streams_progress_and_keeps_complete_outputs(self):
        entries = []
        script = "import sys; print('LOCAL_PROGRESS:50%|2 MiB/s|00:03', flush=True); print('LOCAL_FILE:test.mkv'); print('WARNING: exemple', file=sys.stderr, flush=True)"
        with log_context(sink=entries.append, videoId="abc123def45"):
            stdout, stderr = run(sys.executable, ["-X", "utf8", "-c", script], 5, operation="media.ytdlp")
        self.assertIn("LOCAL_FILE:test.mkv", stdout)
        self.assertIn("WARNING: exemple", stderr)
        progress = next(entry for entry in entries if entry["name"] == "media.download.progress")
        self.assertEqual(progress["fields"]["percent"], "50%")
        self.assertEqual(progress["fields"]["videoId"], "abc123def45")
        self.assertTrue(any(entry["level"] == "warn" for entry in entries))

    def test_media_timeout_is_logged_and_kills_its_child(self):
        entries = []
        with log_context(sink=entries.append):
            with self.assertRaises(subprocess.TimeoutExpired):
                run(sys.executable, ["-c", "import time; time.sleep(10)"], 0.2)
        self.assertEqual(entries[-1]["name"], "media.tool.failed")
        self.assertEqual(entries[-1]["fields"]["errorType"], "TimeoutExpired")

    def test_gpu_snapshot_reads_the_mps_budget_and_allocations(self):
        mps = SimpleNamespace(recommended_max_memory=lambda: 24, driver_allocated_memory=lambda: 16,
                              current_allocated_memory=lambda: 10)
        self.assertEqual(gpu_fields(SimpleNamespace(mps=mps), "mps"), {
            "vramFreeBytes": 8, "vramTotalBytes": 24, "vramAllocatedBytes": 10, "vramReservedBytes": 16,
        })
