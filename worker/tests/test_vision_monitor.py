from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eventlog import log_context
from vision_monitor import GenerationProgress, check_vision_memory, physical_free_memory


class VisionMonitorTests(unittest.TestCase):
    def test_prompt_is_not_counted_and_first_token_latency_is_reported(self):
        events = []
        with log_context(sink=events.append), patch("vision_monitor.time.perf_counter") as clock:
            clock.return_value = 100
            monitor = GenerationProgress()
            monitor.put(SimpleNamespace(shape=(1, 1874)))
            self.assertEqual(monitor.fields(), {"generatedTokens": 0, "waitingForFirstToken": True})
            clock.return_value = 115
            monitor.put(SimpleNamespace(shape=(1,)))
            clock.return_value = 117
            monitor.put(SimpleNamespace(shape=(1,)))
            self.assertEqual(monitor.fields()["generatedTokens"], 2)
            self.assertEqual(monitor.fields()["firstTokenMs"], 15000)
            self.assertEqual(monitor.fields()["tokensPerSecond"], 0.5)
            clock.return_value = 120
            self.assertEqual(monitor.fields()["lastTokenAgeMs"], 3000)
            monitor.end()
        self.assertEqual([e["name"] for e in events],
                         ["vision.generate.first_token", "vision.generate.stream_complete"])
        self.assertNotIn("text", events[0]["fields"])

    def test_memory_guard_rejects_pressure_before_loading_weights(self):
        config = {"visionModel": "Qwen/Qwen3.8-27B", "visionQuantization": "nf4", "device": "cuda:0"}
        gpu = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (14 * 1024 ** 3, 24 * 1024 ** 3)))
        with log_context(sink=lambda entry: None):
            with self.assertRaisesRegex(RuntimeError, "14.0 Gio"):
                check_vision_memory(gpu, config)
            gpu.cuda.mem_get_info = lambda device: (21 * 1024 ** 3, 24 * 1024 ** 3)
            check_vision_memory(gpu, config)

    def test_windows_physical_pressure_is_not_hidden_by_cuda_budget(self):
        config = {"visionModel": "Qwen/Qwen3.8-27B", "visionQuantization": "nf4", "device": "cuda:0"}
        gpu = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda device: (22 * 1024 ** 3, 24 * 1024 ** 3),
            get_device_properties=lambda device: SimpleNamespace(uuid="test")))
        with patch("vision_monitor.subprocess.run", return_value=SimpleNamespace(stdout="14336\n")) as query:
            self.assertEqual(physical_free_memory(gpu, "cuda:0"), 14 * 1024 ** 3)
            self.assertIn("--id=GPU-test", query.call_args.args[0])
            with log_context(sink=lambda entry: None), self.assertRaisesRegex(RuntimeError, "14.0 Gio"):
                check_vision_memory(gpu, config)
        with patch("vision_monitor.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(physical_free_memory(gpu, "cuda:0"))

    def test_9b_uses_its_own_budget(self):
        config = {"visionModel": "Qwen/Qwen3.5-9B", "visionQuantization": "nf4", "device": "cuda:0"}
        gpu = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (12 * 1024 ** 3, 24 * 1024 ** 3)))
        with log_context(sink=lambda entry: None):
            check_vision_memory(gpu, config)
            gpu.cuda.mem_get_info = lambda device: (8 * 1024 ** 3, 24 * 1024 ** 3)
            with self.assertRaisesRegex(RuntimeError, "10 Gio"):
                check_vision_memory(gpu, config)


if __name__ == "__main__":
    unittest.main()
