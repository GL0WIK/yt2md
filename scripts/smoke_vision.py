"""Run one real video group with default settings, without changing production checkpoints."""
from pathlib import Path
import argparse
import json
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
from common import read_json, atomic_json, within_project
from main import local_environment
from runtime import LocalRuntime, model_locks, gpu_fields, IncompleteVisualOutput
from timestamps import audio_context, utterances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--backend", choices=["llama_cpp", "transformers"], default="llama_cpp")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", args.video_id) or args.group_index < 0:
        parser.error("Identifiant vidéo ou index de groupe invalide")
    config = {"modelsDir": str(ROOT / "models"), "device": "cuda:0",
              "asrModel": "Qwen/Qwen3-ASR-1.7B-hf", "asrAlignerModel": "Qwen/Qwen3-ForcedAligner-0.6B-hf", "asrLanguage": "fr", "visionModel": "Qwen/Qwen3.5-9B",
              "visionQuantization": "nf4", "fps": 1, "groupFrames": 7,
              "visionBackend": args.backend, "nativeVisionDir": str(ROOT / "models/native-vision"),
              "cacheDir": str(ROOT / ".smoke-tmp/vision/runtime"), "ffmpegBinary": "ffmpeg",
              "audioChunkSeconds": 60, "audioOverlapSeconds": 1, "frameMaxSide": 1280,
              "visionMaxPixels": 393216, "visionMaxNewTokens": args.max_new_tokens}
    local_environment(config)
    locks = model_locks(config)
    directory = within_project(ROOT / "cache" / args.video_id)
    manifest = read_json(directory / "media/frames.json")
    source = read_json(directory / "source.json")
    if not manifest or not source:
        raise RuntimeError("Cette vidéo ne dispose pas encore de médias préparés.")
    # The transcript key also covers the title context: take the most recent transcript of this video.
    folders = sorted((path for path in (directory / "asr").glob("*") if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not folders:
        raise RuntimeError("Transcription en cache absente : analyser d’abord cette vidéo.")
    transcript = []
    for path in sorted(folders[-1].glob("*.json")):
        record = read_json(path)
        entries, _ = utterances(record["result"], record["window"])
        transcript.extend(entries)
    if not transcript:
        raise RuntimeError("Transcription en cache vide pour cette vidéo.")
    group = manifest["frames"][args.group_index * 7:(args.group_index + 1) * 7]
    if not group:
        raise RuntimeError("Groupe absent de la vidéo.")
    paths = [within_project(directory / "media" / frame["path"]) for frame in group]
    context = audio_context(transcript, group[0]["startMs"], group[-1]["endMs"])
    runtime = LocalRuntime(config, locks)
    try:
        runtime.load("vision")
        started = time.perf_counter()
        result = runtime.observe(paths, group, context, len(manifest["frames"]))
        import torch
        receipt = {"model": config["visionModel"], "revision": locks["vision"]["revision"],
                   "backend": args.backend,
                   "videoId": args.video_id, "groupIndex": args.group_index,
                   "elapsedSeconds": round(time.perf_counter() - started, 3),
                   "frameCount": len(group), "audioContextEntries": len(context),
                   **gpu_fields(torch, config["device"]), "result": result}
        atomic_json(ROOT / ".smoke-tmp/vision/receipt.json", receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "result"}))
    except IncompleteVisualOutput as error:
        atomic_json(ROOT / ".smoke-tmp/vision/incomplete.json", {"model": config["visionModel"], "raw": error.raw})
        raise
    finally:
        runtime.unload()


if __name__ == "__main__":
    main()
