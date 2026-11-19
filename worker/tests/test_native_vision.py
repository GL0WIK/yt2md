from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_vision import NativeVision, request_payload
from runtime import generation_schema
from pipeline import stage_keys


class NativeVisionTests(unittest.TestCase):
    def test_loading_model_503_is_retried_before_ready(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root) as cache:
            config = {"cacheDir": cache, "device": "cuda:0", "visionMaxPixels": 393216}
            lock = {"binary": "unused", "model": "unused", "projector": "unused", "engine": "test"}
            engine = NativeVision(config, lock)
            process = MagicMock(pid=123)
            process.poll.return_value = None
            try:
                with patch("native_vision.subprocess.Popen", return_value=process), \
                     patch.object(engine, "request", side_effect=[urllib.error.HTTPError(
                         "http://127.0.0.1/health", 503, "Loading model", {}, None), {"status": "ok"}]) as request, \
                     patch("native_vision.time.sleep") as sleep:
                    engine.start()
                    self.assertEqual(request.call_count, 2)
                    sleep.assert_called_once_with(0.2)
                    command = __import__("native_vision").subprocess.Popen.call_args.args[0]
                    self.assertEqual(command[command.index("--reasoning") + 1], "off")
                    # The key is passed through a scratch file removed once the engine is ready.
                    self.assertNotIn("--api-key", command)
                    self.assertNotIn(engine.key, command)
                    self.assertFalse(Path(command[command.index("--api-key-file") + 1]).exists())
            finally:
                engine.close()
            process.terminate.assert_called_once()
            self.assertFalse(engine.scratch.exists())

    def test_video_context_and_strict_json_are_preserved(self):
        frames = [{"startMs": 14000, "endMs": 15000, "ptsMs": 14000}]
        context = [{"startMs": 14000, "endMs": 14500, "text": "paroles inchangées"}]
        schema = generation_schema()
        payload = request_payload("encoded", frames, context, schema, 1024)
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "input_video")
        self.assertIn("paroles inchangées", payload["messages"][1]["content"][1]["text"])
        self.assertIn('14000', payload["messages"][1]["content"][1]["text"])
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], schema)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertFalse(payload["cache_prompt"])

    def test_backend_change_keeps_asr_cache_and_invalidates_only_vision(self):
        config = {"asrModel": "asr/model", "asrLanguage": "fr", "visionModel": "vision/model", "audioChunkSeconds": 60,
                  "audioOverlapSeconds": 1, "fps": 1, "groupFrames": 7, "frameMaxSide": 1280,
                  "visionMaxPixels": 393216, "visionMaxNewTokens": 1024, "visionQuantization": "nf4"}
        locks = {role: {"modelId": config[key], "revision": "abc"} for role, key in
                 [("asr", "asrModel"), ("aligner", "asrModel"), ("vision", "visionModel")]}
        with patch("pipeline.packages", return_value={key: "1" for key in
                   ["torch", "transformers", "silero-vad", "qwen-vl-utils", "lm-format-enforcer"]}):
            old_asr, old_vision = stage_keys("source", config, locks)
            new_asr, new_vision = stage_keys("source", {**config, "visionBackend": "llama_cpp"}, locks)
            self.assertEqual(new_asr, old_asr)
            self.assertNotEqual(new_vision, old_vision)


class NativeDownloadTests(unittest.TestCase):
    def build(self, folder, members):
        import io, tarfile
        archive = Path(folder) / "engine.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, kind, mode, target in members:
                info = tarfile.TarInfo(name)
                info.mode = mode
                if kind == "file":
                    info.size = 2
                    tar.addfile(info, io.BytesIO(b"#!"))
                else:
                    info.type, info.linkname = tarfile.SYMTYPE, target
                    tar.addfile(info)
        return archive

    def test_engine_is_the_pinned_macos_arm64_metal_build(self):
        import native_download
        self.assertEqual(native_download.FILES["llama-b11576-bin-macos-arm64.tar.gz"],
                         "e46155b4dff22645df24cd46888e96fc46a417467b0c22983e58eae034fda81f")
        self.assertFalse([name for name in native_download.FILES if "win" in name or "cudart" in name])
        self.assertEqual(set(native_download.FILES), set(native_download.FILE_BYTES))
        self.assertEqual(native_download.BINARY, "bin/llama-b11576/llama-server")

    def test_extraction_keeps_the_executable_bit_and_relative_library_links(self):
        import os
        from native_download import extract_engine
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root) as folder:
            archive = self.build(folder, [("llama-b1/llama-server", "file", 0o755, None),
                                          ("llama-b1/libggml.0.26.0.dylib", "file", 0o755, None),
                                          ("llama-b1/libggml.0.dylib", "link", 0o755, "libggml.0.26.0.dylib")])
            target = Path(folder) / "bin"
            extract_engine(archive, target)
            self.assertTrue(os.access(target / "llama-b1/llama-server", os.X_OK))
            self.assertEqual((target / "llama-b1/libggml.0.dylib").resolve(), (target / "llama-b1/libggml.0.26.0.dylib").resolve())

    def test_extraction_refuses_members_or_links_outside_the_engine_folder(self):
        from native_download import extract_engine
        root = Path(__file__).resolve().parents[2]
        for member in [("../evil", "file", 0o644, None), ("llama-b1/lib.dylib", "link", 0o755, "../../outside")]:
            with tempfile.TemporaryDirectory(dir=root) as folder:
                archive = self.build(folder, [member])
                with self.assertRaises(RuntimeError):
                    extract_engine(archive, Path(folder) / "bin")
                self.assertFalse((Path(folder) / "evil").exists())
