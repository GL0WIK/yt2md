"""Real Qwen3-ASR + forced aligner GPU check on existing local audio; never modifies production checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
from common import atomic_json, read_json, within_project
from main import local_environment
from media import slice_wave
from runtime import LocalRuntime
from timestamps import utterances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="WAV mono 16 kHz existant dans le dossier du projet")
    parser.add_argument("--start", type=int, default=0, help="Début de l’extrait, en secondes")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--language", default="fr", help="Code ISO ou auto")
    parser.add_argument("--context", default="", help="Contexte facultatif (titre, noms propres)")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 200 or args.start < 0:
        parser.error("--seconds doit être entre 1 et 200, --start positif")
    model_dir = within_project(args.models_dir)
    lock = read_json(model_dir / "models.lock.json")
    if not lock or any(role not in lock.get("models", {}) for role in ("asr", "aligner")):
        raise SystemExit("Modèles audio absents. Exécuter npm run models:download -- asr aligner.")
    # Only the audio models are needed: the vision weights may be absent with the native llama.cpp engine.
    locks = {role: {**lock["models"][role], "path": str(within_project(lock["models"][role]["path"]))}
             for role in ("asr", "aligner")}
    config = {"modelsDir": str(model_dir), "device": args.device, "asrLanguage": args.language,
              "asrModel": locks["asr"]["modelId"], "asrAlignerModel": locks["aligner"]["modelId"],
              "visionQuantization": "bf16"}
    local_environment(config)
    destination = within_project(ROOT / ".smoke-tmp/asr")
    destination.mkdir(parents=True, exist_ok=True)
    audio = destination / "sample.wav"
    start_ms, end_ms = args.start * 1000, (args.start + args.seconds) * 1000
    slice_wave(within_project(args.audio), audio, start_ms, end_ms)
    runtime = LocalRuntime(config, locks)
    try:
        runtime.load("asr")
        started = time.perf_counter()
        result = runtime.transcribe(audio, context=args.context)
        elapsed = time.perf_counter() - started
        window = {"index": 0, "startMs": start_ms, "endMs": end_ms, "coreStartMs": start_ms, "coreEndMs": end_ms}
        entries, words = utterances(result, window)
        if not words:
            raise RuntimeError("Aucune parole horodatée dans cet extrait ; choisir un autre audio.")
        receipt = {"model": config["asrModel"], "aligner": config["asrAlignerModel"], "language": args.language,
                   "revision": runtime.locks["asr"]["revision"], "alignerRevision": runtime.locks["aligner"]["revision"],
                   "detectedLanguages": result.get("languages", []),
                   "startSeconds": args.start, "audioSeconds": args.seconds, "elapsedSeconds": round(elapsed, 3),
                   "speedVsRealtime": round(args.seconds / elapsed, 2), "wordCount": len(words),
                   "text": result["text"], "words": words, "utterances": entries}
        atomic_json(destination / "receipt.json", receipt)
        print({key: receipt[key] for key in ("model", "language", "audioSeconds", "elapsedSeconds", "speedVsRealtime", "wordCount")})
        print(result["text"])
    finally:
        runtime.unload()


if __name__ == "__main__":
    main()
