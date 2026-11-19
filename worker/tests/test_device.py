from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT_ROOT
from runtime import doctor


class AppleSiliconDoctorTests(unittest.TestCase):
    def test_doctor_sees_the_mps_gpu_and_asks_only_for_models(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as models:
            config = {"device": "mps", "asrCompile": False, "modelsDir": models, "asrModel": "Qwen/Qwen3-ASR-1.7B-hf",
                      "asrAlignerModel": "Qwen/Qwen3-ForcedAligner-0.6B-hf", "visionModel": "Qwen/Qwen3.5-9B",
                      "visionBackend": "llama_cpp", "nativeVisionDir": models}
            report = doctor(config)
        self.assertNotIn("bitsandbytes", report["packages"])
        self.assertIsNotNone(report["gpu"], report["issues"])
        self.assertTrue(report["gpu"]["name"].startswith("Apple"))
        self.assertGreater(report["gpu"]["totalVramBytes"], 0)
        self.assertEqual(report["issues"], ["Poids asr absents pour Qwen/Qwen3-ASR-1.7B-hf. Exécuter npm run models:download."])


if __name__ == "__main__":
    unittest.main()
