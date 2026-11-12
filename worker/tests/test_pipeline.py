from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT_ROOT, atomic_json, read_json, timestamp, within_project
from media import audio_windows
from pipeline import Pipeline, cleanup, stage_keys
from runtime import ASR_CONTEXT_MAX_CHARS, asr_context, asr_language, asr_token_budget, video_messages, validate_visual
from timestamps import SAMPLE_RATE, align_words, audio_context, first_pause, is_unpunctuated, speech_segments, timed_words, utterances


def recognized(*words):
    return {"text": " ".join(text.strip() for text, _, _ in words),
            "words": [{"text": text, "start": start, "end": end} for text, start, end in words]}


class WordTimestampsTests(unittest.TestCase):
    def window(self, duration=30000, start=0):
        return {"index": 0, "startMs": start, "endMs": start + duration, "coreStartMs": start, "coreEndMs": start + duration}

    def test_word_times_are_offset_by_the_window_and_keep_case_and_punctuation(self):
        result = recognized((" Bonjour,", 9.1, 9.5), (" monde", 9.8, 10.2), (" encore!", 10.4, 10.99), ("  ", 11, 11))
        words = timed_words(result, self.window(start=180000))
        self.assertEqual([w["text"] for w in words], ["Bonjour,", "monde", "encore!"])
        self.assertEqual([w["startMs"] for w in words], [189100, 189800, 190400])
        self.assertEqual([w["endMs"] for w in words], [189500, 190200, 190990])

    def test_rejects_missing_words_malformed_output_and_out_of_bounds_times(self):
        for result in ({}, {"words": "bonjour"}, {"words": [5]}, {"words": [{"text": "bonjour"}]},
                       recognized(("bonjour", -1, 0.1)), recognized(("bonjour", 0.2, 0.1)),
                       recognized(("bonjour", 0.1, 2)), recognized(("bonjour", float("nan"), 0.2)),
                       recognized((" bonjour", 0.2, 0.5), (" monde", 0.1, 0.7))):
            with self.subTest(result=result), self.assertRaises(ValueError):
                timed_words(result, self.window(duration=1000))

    def test_chunk_ownership_and_context_use_real_times(self):
        window = {"index": 1, "startMs": 9000, "endMs": 21000, "coreStartMs": 10000, "coreEndMs": 20000}
        entries, words = utterances(recognized(("avant", 0, 0.8), (" bonjour", 1, 1.25), (" monde", 1.25, 1.5)), window)
        self.assertEqual(len(words), 3)
        self.assertEqual(entries[0]["text"], "bonjour monde")
        self.assertEqual(entries[0]["chunkIndex"], 2)
        self.assertEqual(entries[0]["speaker"], "Indéterminé")
        self.assertEqual(audio_context(entries, 10000, 11000), [{"startMs": 10000, "endMs": 10500, "text": "bonjour monde"}])
        self.assertEqual(audio_context(entries, 11000, 12000), [])
        self.assertEqual(timestamp(1250), "00:01")
        self.assertEqual(timestamp(7600, round_end=True), "00:08")

    def test_silence_is_not_exported_as_speech(self):
        self.assertEqual(utterances(recognized(), self.window())[0], [])


class SpeechRecognitionSettingsTests(unittest.TestCase):
    def seconds(self, start, end):
        return {"start": start * SAMPLE_RATE, "end": end * SAMPLE_RATE}

    def test_a_whole_window_is_one_pass_with_silence_trimmed_and_long_speech_split(self):
        regions = [self.seconds(2, 5), self.seconds(6, 20), self.seconds(25, 61)]
        self.assertEqual(speech_segments(regions, 62 * SAMPLE_RATE), [(2 * SAMPLE_RATE, 61 * SAMPLE_RATE)])
        self.assertEqual(speech_segments([self.seconds(0, 300)], 300 * SAMPLE_RATE),
                         [(0, 130 * SAMPLE_RATE), (130 * SAMPLE_RATE, 260 * SAMPLE_RATE), (260 * SAMPLE_RATE, 300 * SAMPLE_RATE)])
        self.assertEqual(speech_segments([], SAMPLE_RATE), [])

    def test_language_is_forced_unless_auto_and_budget_scales_with_audio(self):
        self.assertEqual(asr_language({"asrLanguage": "fr"}), "fr")
        self.assertIsNone(asr_language({"asrLanguage": "auto"}))
        self.assertGreater(asr_token_budget(60), 60 * 4)
        self.assertEqual(asr_token_budget(60, attempt=1), 2 * asr_token_budget(60))

    def test_context_is_title_and_channel_only_when_enabled(self):
        video = {"title": "Test Flashforge Creator 5", "channel": "Les Frères Poulain", "description": "ignorée"}
        self.assertEqual(asr_context(video, {"asrContext": "title"}), "Test Flashforge Creator 5\nLes Frères Poulain")
        self.assertEqual(asr_context(video, {"asrContext": "none"}), "")
        self.assertEqual(asr_context({"title": "x" * 1000, "channel": None}, {}), "x" * ASR_CONTEXT_MAX_CHARS)


def split(token):
    # Same rule as the processor's splitter for space-delimited languages: punctuation is not aligned.
    kept = "".join(char for char in token if char.isalnum() or char == "'")
    return [kept] if kept else []


def stamp(start, end):
    return {"start_time": start, "end_time": end}


class PunctuationRecoveryTests(unittest.TestCase):
    def test_only_long_text_without_any_punctuation_is_flagged(self):
        self.assertTrue(is_unpunctuated("dans un polystyrène aussi bien toutes les têtes " * 3))
        self.assertFalse(is_unpunctuated("Dans un polystyrène, aussi bien toutes les têtes " * 3))
        self.assertFalse(is_unpunctuated("oui oui"))

    def test_cut_is_the_widest_early_gap_between_words(self):
        words = [{"start": 0.0, "end": 0.9}, {"start": 1.0, "end": 2.0}, {"start": 2.6, "end": 3.0},
                 {"start": 3.1, "end": 4.0}, {"start": 20.0, "end": 21.0}, {"start": 22.0, "end": 23.0}]
        self.assertAlmostEqual(first_pause(words, 60.0), 2.3)
        # Gaps after 15 s or leaving less than 5 s of audio are never used.
        self.assertIsNone(first_pause(words[4:], 60.0))
        self.assertIsNone(first_pause(words[:3], 6.0))


class AlignmentMappingTests(unittest.TestCase):
    def test_aligned_units_keep_recognized_punctuation_and_typography(self):
        words = align_words("Salut à tous ! C’est parti, l’imprimante…", [stamp(0.1, 0.3), stamp(0.3, 0.4), stamp(0.4, 0.6),
                                                                       stamp(0.8, 1.0), stamp(1.0, 1.2), stamp(1.3, 1.9)], split)
        self.assertEqual([word["text"] for word in words], ["Salut", "à", "tous !", "C’est", "parti,", "l’imprimante…"])
        self.assertEqual((words[2]["start"], words[2]["end"]), (0.4, 0.6))
        self.assertEqual((words[-1]["start"], words[-1]["end"]), (1.3, 1.9))

    def test_leading_punctuation_joins_the_next_word(self):
        words = align_words("« Bonjour »", [stamp(0.0, 0.5)], split)
        self.assertEqual(words, [{"text": "« Bonjour »", "start": 0.0, "end": 0.5}])

    def test_a_mismatch_between_text_and_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            align_words("deux mots", [stamp(0, 1)], split)
        with self.assertRaises(ValueError):
            align_words("un", [stamp(0, 1), stamp(1, 2)], split)

    def test_windows_cover_full_duration_and_preserve_overlap(self):
        windows = audio_windows(400100, 60, 1)
        self.assertEqual(len(windows), 7)
        self.assertEqual(windows[0]["startMs"], 0)
        self.assertEqual(windows[1]["startMs"], 59000)
        self.assertEqual(windows[-1]["coreEndMs"], 400100)
        self.assertTrue(all(w["endMs"] - w["startMs"] <= 62000 for w in windows))


class VideoContractTests(unittest.TestCase):
    def test_native_video_has_ordered_frames_and_separate_untrusted_audio_context(self):
        root = PROJECT_ROOT / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            frames = [{"startMs": 7000 + i * 1000, "endMs": 8000 + i * 1000, "ptsMs": 7000 + i * 1000} for i in range(7)]
            paths = [Path(temp) / f"{i}.jpg" for i in range(7)]
            context = [{"startMs": 7100, "endMs": 7350, "text": "paroles originales"}]
            messages = video_messages(paths, frames, context, {"fps": 1, "visionMaxPixels": 393216})
            video = messages[1]["content"][0]
            self.assertEqual(video["type"], "video")
            self.assertEqual(len(video["video"]), 7)
            self.assertEqual(video["sample_fps"], 1)
            self.assertIn("paroles originales", messages[1]["content"][1]["text"])
            self.assertIn("pas une preuve visuelle", messages[0]["content"][0]["text"])

    def test_visual_json_rejects_extra_fields_and_missing_evidence_description(self):
        value = {"title": "Plan", "visualDescription": "Écran visible", "screenText": [], "uncertainties": []}
        self.assertEqual(validate_visual(value), value)
        for broken in ({**value, "transcript": "réécrit"}, {**value, "title": ""}, {**value, "screenText": [5]}, {"title": "Plan"}):
            with self.subTest(value=broken), self.assertRaises(ValueError):
                validate_visual(broken)

    def test_runtime_paths_cannot_escape_project_folder(self):
        with self.assertRaises(ValueError):
            within_project(PROJECT_ROOT.parent / "outputs")


class FakeRuntime:
    def __init__(self):
        self.asr_calls = self.vision_calls = 0
        self.fail_at = None
        self.contexts, self.asr_contexts = [], []

    def transcribe(self, audio, attempt=0, context=""):
        self.asr_calls += 1
        self.asr_contexts.append(context)
        return {**recognized(("bonjour", 1, 1.3), (" qwen", 1.3, 1.7)),
                "speechDetected": True, "inputTokens": 0, "outputTokens": 2}

    def observe(self, paths, frames, context, total, attempt=0):
        self.vision_calls += 1
        if self.fail_at == self.vision_calls:
            raise RuntimeError("Interruption du GPU simulée")
        self.contexts.append(copy.deepcopy(context))
        return {"observation": {"title": "Plan", "visualDescription": "Une interface est visible.", "screenText": ["Texte affiché"], "uncertainties": []},
                "inputTokens": 20, "outputTokens": 10}


class RestartablePipelineTests(unittest.TestCase):
    def setUp(self):
        root = PROJECT_ROOT / ".test-tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = {"cacheDir": str(self.directory), "asrModel": "Qwen/Qwen3-ASR-1.7B-hf",
                       "asrAlignerModel": "Qwen/Qwen3-ForcedAligner-0.6B-hf", "asrLanguage": "fr", "asrContext": "title",
                       "visionModel": "Qwen/Qwen3.5-9B",
                       "audioChunkSeconds": 60, "audioOverlapSeconds": 1,
                       "fps": 1, "groupFrames": 7, "frameMaxSide": 1280, "visionMaxPixels": 393216,
                       "visionMaxNewTokens": 1024, "visionQuantization": "nf4", "keepMedia": False}
        self.locks = {role: {"modelId": self.config[key], "revision": "abc", "path": "local"}
                      for role, key in [("asr", "asrModel"), ("aligner", "asrAlignerModel"), ("vision", "visionModel")]}
        self.video = {"id": "abc123def45", "url": "https://www.youtube.com/watch?v=abc123def45", "title": "Démo", "channel": "Chaîne"}
        self.frames = [{"index": i, "path": f"frames/{i}.jpg", "ptsMs": i * 1000, "startMs": i * 1000, "endMs": min((i + 1) * 1000, 15600)} for i in range(16)]
        self.manifest = {"durationMs": 15600, "durationSeconds": 16, "sourceStartSeconds": 0, "audio": "audio.wav", "frames": self.frames}
        self.runtime = FakeRuntime()
        self.progress = []
        self.download = patch("pipeline.media.download", return_value=(self.directory / "source.mkv", {"sha256": "a" * 64}, []))
        self.extract = patch("pipeline.media.extract", return_value=self.manifest)
        self.slice = patch("pipeline.media.slice_wave")
        self.download.start(); self.extract.start(); self.slice.start()
        self.addCleanup(self.download.stop); self.addCleanup(self.extract.stop); self.addCleanup(self.slice.stop)

    def run_pipeline(self):
        return Pipeline(self.config, self.locks, self.runtime, lambda *p: self.progress.append(p)).analyze(self.video)

    def test_resume_keeps_completed_audio_and_video_groups_without_rewriting_transcript(self):
        self.runtime.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "Interruption"):
            self.run_pipeline()
        self.assertEqual(self.runtime.asr_calls, 1)
        self.runtime.fail_at = None
        result = self.run_pipeline()
        self.assertEqual(self.runtime.asr_calls, 1)
        self.assertEqual(self.runtime.vision_calls, 4)  # 2 before crash, 2 remaining afterwards
        self.assertEqual(len(result["visualSegments"]), 3)
        self.assertEqual(result["visualSegments"][-1]["endTimestamp"], "00:16")
        self.assertEqual(result["transcript"][0]["text"], "bonjour qwen")
        # 1000-1700 ms: start rounds down, end rounds up, never 00:01-00:01.
        self.assertEqual((result["transcript"][0]["timestamp"], result["transcript"][0]["endTimestamp"]), ("00:01", "00:02"))
        self.assertEqual(result["localProvenance"]["words"][0]["endMs"], 1300)
        self.assertEqual(result["usage"]["cacheHits"], 2)
        self.assertEqual(self.runtime.contexts[0][0]["text"], "bonjour qwen")
        again = self.run_pipeline()
        self.assertEqual(again["usage"]["totalTokenCount"], 0)
        self.assertEqual(self.runtime.vision_calls, 4)
        self.assertFalse(any(p[0] == "align" for p in self.progress))

    def test_vision_configuration_invalidates_only_vision_cache(self):
        first = self.run_pipeline()
        self.config["visionMaxPixels"] //= 2
        second = self.run_pipeline()
        self.assertNotEqual(first["cacheKey"], second["cacheKey"])
        self.assertEqual(self.runtime.asr_calls, 1)
        self.assertEqual(self.runtime.vision_calls, 6)
        self.locks["asr"]["revision"] = "new"
        self.run_pipeline()
        self.assertEqual(self.runtime.asr_calls, 2)

    def test_cleanup_leaves_checkpoints_and_never_removes_incomplete_work(self):
        with self.assertRaises(ValueError):
            cleanup(self.config, self.video["id"])
        self.run_pipeline()
        media = self.directory / self.video["id"] / "media"
        media.mkdir(exist_ok=True)
        (media / "dummy").write_text("cache", encoding="utf-8")
        cleanup(self.config, self.video["id"])
        self.assertFalse(media.exists())
        self.assertEqual(read_json(self.directory / self.video["id"] / "manifest.json")["status"], "exported")
        self.assertTrue(list((self.directory / self.video["id"] / "asr").rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
