"""Regression: some checkpoints default to endoftext, while the JSON decoder emits im_end."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import LocalRuntime, VISUAL_SCHEMA, generation_schema, parse_visual_output, IncompleteVisualOutput
from eventlog import log_context


class VisionGenerationTests(unittest.TestCase):
    def test_complete_json_at_token_limit_is_accepted_without_repairing_partial_json(self):
        raw = '{"title":"Plan","visualDescription":"Un écran.","screenText":[],"uncertainties":[]}'
        self.assertEqual(parse_visual_output(raw, 1024, 1024)["title"], "Plan")
        with self.assertRaises(IncompleteVisualOutput):
            parse_visual_output(raw[:-1], 1024, 1024)
        with self.assertRaises(ValueError):
            parse_visual_output('{"title":"Plan"}', 1024, 1024)

    def test_retry_budget_is_compact_without_changing_export_validation(self):
        normal, retry = generation_schema(), generation_schema(1)
        self.assertEqual(set(normal["properties"]), set(VISUAL_SCHEMA["properties"]))
        self.assertLess(retry["properties"]["visualDescription"]["maxLength"], normal["properties"]["visualDescription"]["maxLength"])
        self.assertEqual(VISUAL_SCHEMA["properties"]["visualDescription"]["maxLength"], 4000)
        self.assertEqual(VISUAL_SCHEMA["properties"]["screenText"]["maxItems"], 40)

    def test_generation_stops_on_the_same_eos_as_the_constrained_decoder(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Output:
            def __init__(self, generated):
                self.generated = generated

            def __getitem__(self, key):
                return SimpleNamespace(shape=(1, self.generated))

        def generate(**kwargs):
            # Emulate a valid response followed by im_end. Without the explicit
            # chat EOS, the base checkpoint continues to the token limit.
            return Output(20 if kwargs.get("eos_token_id", 248044) == 248046 else kwargs["max_new_tokens"])

        runtime = LocalRuntime({"device": "mps", "visionMaxPixels": 393216, "visionMaxNewTokens": 1024}, {})
        runtime.role = "vision"
        runtime.model = SimpleNamespace(device="mps", generate=generate)
        runtime.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=248046, pad_token_id=248044),
            batch_decode=lambda *a, **kw: ['{"title":"Plan","visualDescription":"Un écran.","screenText":[],"uncertainties":[]}'])
        runtime.tokenizer_data = object()
        torch = SimpleNamespace(inference_mode=nullcontext)
        decoder = SimpleNamespace(tokenizer_data=lambda tokenizer: None, json_prefix=lambda *args: None)
        with patch.dict(sys.modules, {"torch": torch, "json_constraint": decoder}), \
             patch("runtime.video_inputs", return_value=Inputs(input_ids=SimpleNamespace(shape=(1, 10)))), \
             patch("runtime.gpu_fields", return_value={}), log_context(sink=lambda entry: None):
            result = runtime.observe([], [], [], 0)
        self.assertEqual(result["outputTokens"], 20)
        self.assertEqual(result["observation"]["title"], "Plan")
