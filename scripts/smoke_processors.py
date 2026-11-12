"""Download only configs/tokenizers and exercise the real audio/video processors.

No model weights or inference, all downloads and generated files stay in the project folder.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".smoke-tmp/processors"
TEMP.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TEMP / ".hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, str(ROOT / "worker"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoConfig, AutoModelForMultimodalLM, AutoModelForTokenClassification
from transformers.models.qwen3_asr.processing_qwen3_asr import FORCED_ALIGNER_LANGUAGES
from PIL import Image
import torch
from runtime import VISUAL_SCHEMA, video_inputs
from timestamps import align_words
from json_constraint import tokenizer_data, json_prefix

for role, name in [("asr", "Qwen/Qwen3-ASR-1.7B-hf"), ("aligner", "Qwen/Qwen3-ForcedAligner-0.6B-hf"), ("vision", "Qwen/Qwen3.5-9B")]:
    folder = TEMP / role
    snapshot_download(name, local_dir=str(folder), allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"],
                      ignore_patterns=["*fp32*"])
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if role == "asr":
        assert config.__class__ in AutoModelForMultimodalLM._model_mapping
        inputs = processor.apply_transcription_request(audio=[torch.zeros(32000).numpy()], language=["fr"],
                                                       prompt=["Titre de démonstration"])
        prompt = processor.tokenizer.decode(inputs["input_ids"][0])
        # The language is forced by prefilling the answer: the model only writes French text.
        assert prompt.rstrip().endswith("language French<asr_text>"), prompt[-80:]
        assert inputs["input_features"].shape[-1] > 0
        print("Qwen3-ASR : processeur audio, contexte et français imposé validés.", flush=True)
    elif role == "aligner":
        assert config.__class__ in AutoModelForTokenClassification._model_mapping
        assert "French" in FORCED_ALIGNER_LANGUAGES
        text = "Bonjour à tous ! C’est parti."
        inputs, word_lists = processor.prepare_forced_aligner_inputs(audio=[torch.zeros(32000).numpy()], transcript=[text],
                                                                      language=["French"])
        assert word_lists == [["Bonjour", "à", "tous", "Cest", "parti"]], word_lists
        stamps = [{"start_time": i / 10, "end_time": (i + 1) / 10} for i in range(len(word_lists[0]))]
        words = align_words(text, stamps, lambda token: processor.split_words_for_alignment(token, "French"))
        assert [word["text"] for word in words] == ["Bonjour", "à", "tous !", "C’est", "parti."], words
        print("Qwen3-ForcedAligner : découpage français et rattachement de la ponctuation validés.", flush=True)
    else:
        assert config.__class__ in AutoModelForMultimodalLM._model_mapping
        paths = []
        frames = []
        for i in range(7):
            target = TEMP / f"frame-{i}.jpg"
            Image.new("RGB", (320, 180), color=(i * 20, 80, 120)).save(target)
            paths.append(target)
            frames.append({"index": i + 7, "ptsMs": (i + 7) * 1000, "startMs": (i + 7) * 1000, "endMs": (i + 8) * 1000})
        inputs = video_inputs(processor, paths, frames, [{"startMs": 7500, "endMs": 8900, "text": "bonjour"}], 20,
                              {"fps": 1, "visionMaxPixels": 262144})
        assert "pixel_values_videos" in inputs and "video_grid_thw" in inputs
        assert inputs["video_grid_thw"].shape[0] == 1
        data = tokenizer_data(processor.tokenizer)
        prefix = json_prefix(data, VISUAL_SCHEMA, inputs["input_ids"].shape[1])
        target = '{"title":"Plan","visualDescription":"Interface visible","screenText":[],"uncertainties":[]}'
        ids = processor.tokenizer.encode(target, add_special_tokens=False)
        current = inputs["input_ids"][0]
        for token in ids:
            assert token in prefix(0, current), f"Valid JSON token {token} was rejected"
            current = torch.cat([current, torch.tensor([token])])
        assert processor.tokenizer.eos_token_id in prefix(0, current)
        print(f"Qwen : entrée vidéo native validée, grille {inputs['video_grid_thw'].tolist()}, JSON contraint validé.", flush=True)
