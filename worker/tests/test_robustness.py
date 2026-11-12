"""Worker robustness: transient engine failures, partial files, protocol isolation and cache keys."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT_ROOT
import main
import media
import native_download
import native_vision
from native_vision import NativeVision, TransientVisionError, bind_to_worker, cache_identity, release_job, remove_stale_scratch
from pipeline import stage_keys
from runtime import runtime_fingerprint
import test_pipeline
from test_pipeline import FakeRuntime, recognized

WORKER = Path(__file__).resolve().parents[1]
VISION = {"title": "Plan", "visualDescription": "Un écran.", "screenText": [], "uncertainties": []}


def scratch_root():
    root = PROJECT_ROOT / ".test-tmp"
    root.mkdir(exist_ok=True)
    return root


class FlakyRuntime(FakeRuntime):
    def __init__(self, vision_error=None, asr_error=False):
        super().__init__()
        self.vision_error, self.asr_error = vision_error, asr_error
        self.asr_attempts, self.vision_attempts = [], []

    def transcribe(self, audio, attempt=0, context=""):
        self.asr_attempts.append(attempt)
        if self.asr_error and not attempt:
            self.asr_calls += 1
            return {**recognized(("bonjour", float("nan"), 1)), "speechDetected": True}
        return super().transcribe(audio, attempt, context)

    def observe(self, paths, frames, context, total, attempt=0):
        self.vision_attempts.append(attempt)
        if self.vision_error is not None and len(self.vision_attempts) == 1:
            raise self.vision_error
        return super().observe(paths, frames, context, total, attempt)


class TransientRetryTests(unittest.TestCase):
    setUp, run_pipeline = test_pipeline.RestartablePipelineTests.setUp, test_pipeline.RestartablePipelineTests.run_pipeline

    def test_engine_timeouts_and_empty_answers_are_retried_once(self):
        for error in (TimeoutError("timed out"), TransientVisionError("HTTP 400"), TypeError("None content")):
            with self.subTest(error=type(error).__name__):
                self.runtime = FlakyRuntime(vision_error=error)
                self.config["groupFrames"] += 1  # new vision key: no cache hit between subtests
                result = self.run_pipeline()
                self.assertEqual(self.runtime.vision_attempts[:2], [0, 1])
                self.assertEqual(len(result["visualSegments"]), 2)

    def test_gpu_failure_is_not_retried(self):
        self.runtime = FlakyRuntime(vision_error=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.run_pipeline()
        self.assertEqual(self.runtime.vision_attempts, [0])

    def test_asr_retry_is_a_different_decode(self):
        self.runtime = FlakyRuntime(asr_error=True)
        result = self.run_pipeline()
        self.assertEqual(self.runtime.asr_attempts, [0, 1])
        self.assertEqual(result["transcript"][0]["text"], "bonjour qwen")


class NativeEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=scratch_root())
        self.addCleanup(self.temp.cleanup)
        self.engine = NativeVision({"cacheDir": self.temp.name, "device": "cuda:0", "visionMaxPixels": 393216,
                                    "visionMaxNewTokens": 512}, {"engine": "test"})
        self.addCleanup(self.engine.close)
        self.engine.child = MagicMock()
        self.engine.child.poll.return_value = None
        self.engine.encode = MagicMock(return_value="encoded")
        self.frames = [{"startMs": 0, "endMs": 1000, "ptsMs": 0}]

    def test_timeout_restarts_the_single_slot_engine(self):
        with patch.object(self.engine, "request", side_effect=TimeoutError("timed out")), \
             patch.object(self.engine, "restart") as restart, self.assertRaises(TransientVisionError):
            self.engine.observe([], self.frames, [], 0)
        restart.assert_called_once()

    def test_dead_engine_is_restarted_before_the_request(self):
        self.engine.child.poll.return_value = 1
        response = {"usage": {"completion_tokens": 10, "prompt_tokens": 5},
                    "choices": [{"message": {"content": __import__("json").dumps(VISION)}}]}
        with patch.object(self.engine, "request", return_value=response), patch.object(self.engine, "restart") as restart:
            self.assertEqual(self.engine.observe([], self.frames, [], 0)["observation"], VISION)
        restart.assert_called_once()

    def test_missing_content_is_an_invalid_answer(self):
        response = {"usage": {"completion_tokens": 0, "prompt_tokens": 5}, "choices": [{"message": {"content": None}}]}
        with patch.object(self.engine, "request", return_value=response), self.assertRaises(ValueError):
            self.engine.observe([], self.frames, [], 0)

    def test_stale_scratch_folders_are_removed_but_live_ones_kept(self):
        cache = Path(self.temp.name)
        old, fresh = cache / "native-vision-old", cache / "native-vision-fresh"
        for folder in (old, fresh):
            folder.mkdir()
            (folder / "engine.log").write_text("log", encoding="utf-8")
        past = native_vision.PROCESS_STARTED - 3600
        for item in (old / "engine.log", old):
            os.utime(item, (past, past))
        remove_stale_scratch(cache)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(self.engine.scratch.exists())
        self.assertFalse(list(cache.glob("native-vision-stale-*")))

    @unittest.skipUnless(os.name == "nt", "Job Objects are Windows-only")
    def test_job_object_kills_the_engine_with_the_worker(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            job = bind_to_worker(child)
            self.assertTrue(job)
            release_job(job)  # what the OS does when the worker dies
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


class DownloadTests(unittest.TestCase):
    def test_partial_or_corrupt_download_never_becomes_the_target(self):
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            target, cached = Path(temp) / "engine.zip", Path(temp) / "absent.zip"
            good = b"complete archive"
            digest = hashlib.sha256(good).hexdigest()

            def interrupted(url, path):
                Path(path).write_bytes(good[:5])
                raise OSError("connexion interrompue")
            with patch("native_download.urllib.request.urlretrieve", side_effect=interrupted), self.assertRaises(OSError):
                native_download.fetch("engine.zip", digest, target, cached, True, None)
            self.assertEqual(list(Path(temp).iterdir()), [])
            with patch("native_download.urllib.request.urlretrieve", side_effect=lambda url, path: Path(path).write_bytes(b"bad")), \
                 self.assertRaisesRegex(RuntimeError, "Empreinte"):
                native_download.fetch("engine.zip", digest, target, cached, True, None)
            self.assertEqual(list(Path(temp).iterdir()), [])
            with patch("native_download.urllib.request.urlretrieve", side_effect=lambda url, path: Path(path).write_bytes(good)):
                self.assertTrue(native_download.fetch("engine.zip", digest, target, cached, True, None))
            self.assertEqual(list(Path(temp).iterdir()), [target])


class MediaTests(unittest.TestCase):
    def details(self, duration_ms, video_stream):
        return {"durationMs": duration_ms, "streams": [{"codec_type": "audio"}, {"codec_type": "video", **video_stream}]}

    def test_video_stream_ending_before_audio_is_tolerated_only_when_explained(self):
        media.check_frame_coverage(101, self.details(100200, {}))
        # Matroska tag: video ends at 98.5 s, audio at 100.2 s; frames 0..98 are all there.
        media.check_frame_coverage(99, self.details(100200, {"tags": {"DURATION": "00:01:38.500000000"}}))
        media.check_frame_coverage(99, self.details(100200, {"duration": "98.5"}))
        for count, stream in ((98, {"duration": "98.5"}),          # a frame is missing inside the video stream
                              (97, {"duration": "96.5"}),          # more than 2 tail seconds without image
                              (99, {}),                            # unknown video duration
                              (102, {"duration": "100.2"})):       # too many frames
            with self.subTest(count=count, stream=stream), self.assertRaises(RuntimeError):
                media.check_frame_coverage(count, self.details(100200, stream))

    def test_interrupted_audio_extraction_keeps_previous_file(self):
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            audio = Path(temp) / "audio.wav"
            audio.write_bytes(b"previous")

            def failing(binary, args, timeout, **kwargs):
                Path(args[-1]).write_bytes(b"trunc")
                raise RuntimeError("ffmpeg interrompu")
            config = {"frameMaxSide": 1280, "ffmpegBinary": "ffmpeg", "downloadTimeoutMs": 1000}
            with patch("media.probe", return_value=self.details(1000, {})), patch("media.run", side_effect=failing), \
                 self.assertRaises(RuntimeError):
                media.extract(Path(temp) / "source.mkv", Path(temp), config, "hash")
            self.assertEqual(audio.read_bytes(), b"previous")
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ["audio.wav"])


class ProtocolTests(unittest.TestCase):
    def test_native_writes_to_fd1_go_to_stderr(self):
        script = ("import os, sys; sys.path.insert(0, sys.argv[1]); import main; out = main.protocol_output(); "
                  "os.write(1, b'native noise\\n'); print('python noise'); out.write('{\"id\": 1}\\n'); out.flush()")
        done = subprocess.run([sys.executable, "-c", script, str(WORKER)], capture_output=True, timeout=60)
        self.assertEqual(done.stdout, b'{"id": 1}\n')
        self.assertIn(b"native noise", done.stderr)
        self.assertIn(b"python noise", done.stderr)

    def test_path_is_prepended_once(self):
        with patch.dict(os.environ, {"PATH": os.pathsep.join(["C:/autre", "/usr/bin"])}):
            config = {"modelsDir": "models"}
            main.local_environment(config)
            once = os.environ["PATH"]
            main.local_environment(config)
            self.assertEqual(os.environ["PATH"], once)
            self.assertTrue(once.startswith(str(Path(sys.executable).parent)))


class CacheKeyTests(unittest.TestCase):
    config = {"asrModel": "asr/model", "asrLanguage": "fr", "visionModel": "vision/model", "audioChunkSeconds": 60,
              "audioOverlapSeconds": 1, "fps": 1, "groupFrames": 7, "frameMaxSide": 1280, "visionMaxPixels": 393216,
              "visionMaxNewTokens": 1024, "visionQuantization": "nf4", "visionBackend": "llama_cpp",
              "keepMedia": False, "downloadTimeoutMs": 1000, "device": "cuda:0"}

    def locks(self, native_dir):
        native = {"engine": "b1", "modelRevision": "r", **{name: str(PROJECT_ROOT / native_dir / name)
                                                          for name in ("binary", "model", "projector")}}
        return {"asr": {"modelId": "asr/model", "revision": "a"}, "aligner": {"modelId": "aligner/model", "revision": "l"},
                "vision": {"modelId": "vision/model", "revision": "v", "native": native}}

    def test_compilation_does_not_invalidate_transcripts(self):
        keys = self.keys(self.locks("models/native"))
        with patch.dict(self.config, {"asrCompile": True}):
            self.assertEqual(self.keys(self.locks("models/native")), keys)

    def keys(self, locks):
        with patch("pipeline.packages", return_value={key: "1" for key in
                   ["torch", "transformers", "silero-vad", "qwen-vl-utils", "lm-format-enforcer"]}):
            return stage_keys("source", self.config, locks)

    def test_keys_use_project_relative_paths_and_decoding_settings(self):
        identity = cache_identity(self.locks("models/native")["vision"]["native"])
        self.assertEqual(identity["binary"], "models/native/binary")
        self.assertIn("-np", identity["serverArgs"])
        self.assertEqual(identity["sampling"], native_vision.SAMPLING)
        asr, vision = self.keys(self.locks("models/native"))
        self.assertEqual(self.keys(self.locks("models/native")), (asr, vision))
        self.assertNotEqual(self.keys(self.locks("models/other"))[1], vision)
        with patch("pipeline.ASR_SETTINGS", {"vad": {"speech_pad_ms": 100}}):
            changed_asr, changed_vision = self.keys(self.locks("models/native"))
        self.assertNotEqual(changed_asr, asr)
        self.assertNotEqual(changed_vision, vision)
        with patch("native_vision.SERVER_ARGS", ["-c", "4096"]):
            self.assertEqual(self.keys(self.locks("models/native"))[0], asr)
            self.assertNotEqual(self.keys(self.locks("models/native"))[1], vision)

    def test_non_model_settings_do_not_reload_models(self):
        locks = self.locks("models/native")
        key = runtime_fingerprint(self.config, locks)
        self.assertEqual(runtime_fingerprint({**self.config, "keepMedia": True, "downloadTimeoutMs": 5}, locks), key)
        self.assertNotEqual(runtime_fingerprint({**self.config, "visionQuantization": "bf16"}, locks), key)

if __name__ == "__main__":
    unittest.main()


class CompileTests(unittest.TestCase):
    def test_only_the_asr_decoder_is_compiled_with_eager_rounding(self):
        from types import SimpleNamespace
        import runtime
        inductor = SimpleNamespace(emulate_precision_casts=False)
        fake_torch = SimpleNamespace(_dynamo=SimpleNamespace(config=SimpleNamespace(recompile_limit=8)),
                                     _inductor=SimpleNamespace(config=inductor),
                                     compile=MagicMock(side_effect=lambda function, **options: ("compiled", function, options)))
        engine = runtime.LocalRuntime.__new__(runtime.LocalRuntime)
        engine.config = {"asrCompile": True}
        decoder = SimpleNamespace(forward="decoder.forward")
        engine.model = SimpleNamespace(forward="asr.forward", model=SimpleNamespace(language_model=decoder,
                                                                                     audio_tower=SimpleNamespace(forward="encoder")))
        engine.aligner = SimpleNamespace(forward="aligner.forward")
        modules = {"torch": fake_torch, "torch._inductor": fake_torch._inductor, "torch._inductor.config": inductor}
        with patch.object(runtime, "compile_available", return_value=True), patch.dict(sys.modules, modules), \
                patch.object(runtime, "event"):
            engine._compile_asr(fake_torch)
        # Whole forwards share one Dynamo cache through Transformers' wrapper; the encoder recompiles per length.
        self.assertEqual(decoder.forward, ("compiled", "decoder.forward", {"dynamic": True}))
        self.assertEqual((engine.model.forward, engine.aligner.forward), ("asr.forward", "aligner.forward"))
        self.assertEqual(engine.model.model.audio_tower.forward, "encoder")
        self.assertTrue(inductor.emulate_precision_casts)
        self.assertGreaterEqual(fake_torch._dynamo.config.recompile_limit, 16)


class AudioTrackWarningTests(unittest.TestCase):
    def test_single_language_needs_no_original_label(self):
        from media import audio_track_warnings
        track = {"format_id": "251", "acodec": "opus", "language": "fr", "format_note": "medium"}
        self.assertEqual(audio_track_warnings({"formats": [track]}, [track]), [])

    def test_dubbed_video_without_original_track_is_reported(self):
        from media import audio_track_warnings
        dubbed = {"format_id": "251-1", "acodec": "opus", "language": "en", "format_note": "English, medium"}
        original = {"format_id": "251-0", "acodec": "opus", "language": "fr", "format_note": "French (original), medium"}
        self.assertEqual(audio_track_warnings({"formats": [dubbed, original]}, [original]), [])
        self.assertEqual(len(audio_track_warnings({"formats": [dubbed, original]}, [dubbed])), 1)


class FrameDecoderTests(unittest.TestCase):
    def test_gpu_decoding_falls_back_to_the_processor(self):
        import media
        calls = []

        def fake_extract(source, frames_dir, side, config, progress, *, gpu):
            calls.append(gpu)
            if gpu:
                raise RuntimeError("ffmpeg.exe a échoué (1) : No device available for decoder")
            return ""

        with patch.object(media, "extract_frames", fake_extract), \
                patch.object(media, "probe", return_value={"durationMs": 0, "durationSeconds": 0}), \
                patch.object(media, "run", return_value=("", "")), \
                patch.object(media.os, "replace"):
            with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
                with self.assertRaises(RuntimeError):  # No frame was produced by the fake extraction.
                    media.extract(Path(directory) / "source.mkv", Path(directory) / "media",
                                  {"ffmpegBinary": "ffmpeg", "frameMaxSide": 1280, "downloadTimeoutMs": 1000}, "hash")
        self.assertEqual(calls, [True, False])
