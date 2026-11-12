"""Restartable stages. Cache records contain millisecond times, never corpus extensions."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import shutil
import time
import urllib.error

from timestamps import audio_context, utterances
from common import PIPELINE_VERSION, PROMPT_VERSION, atomic_json, file_sha256, fingerprint, read_json, timestamp, within_project
import media
from native_vision import TransientVisionError, cache_identity
from runtime import ASR_SETTINGS, VISION_SAMPLING, asr_context, validate_visual
from eventlog import event, log_context, stage

# Invalid output or a transient engine failure (timeout, refused or empty response): one retry per group.
VISION_RETRYABLE = (ValueError, KeyError, TypeError, TimeoutError, urllib.error.URLError, TransientVisionError)

# Part of the audio cache key: changing segmentation or decoding recomputes transcripts.
ASR_DECODER = "qwen3-asr-vad-forced-aligner-v1"


def packages() -> dict:
    result = {}
    for name in ("torch", "transformers", "qwen-vl-utils", "lm-format-enforcer", "silero-vad"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "uninstalled"
    return result


def stage_keys(source_hash: str, config: dict, locks: dict, context: str = "") -> tuple[str, str]:
    revisions = {role: {"id": record["modelId"], "revision": record["revision"]} for role, record in locks.items()}
    versions = packages()
    asr = fingerprint({"pipeline": PIPELINE_VERSION, "source": source_hash, "model": revisions["asr"],
                       "aligner": revisions["aligner"], "context": fingerprint(context),
                       # ASR_COMPILE is deliberately absent: compiled kernels only round differently (2 words in
                       # ~900 on the benchmark), so compiled and eager transcripts share their cache.
                       "chunkSeconds": config["audioChunkSeconds"], "language": config["asrLanguage"],
                       "overlapSeconds": config["audioOverlapSeconds"], "decoder": ASR_DECODER,
                       "decoderSettings": fingerprint(ASR_SETTINGS),
                       "torch": versions["torch"], "transformers": versions["transformers"], "vad": versions["silero-vad"]})
    vision = fingerprint({"asr": asr, "model": revisions["vision"], "promptVersion": PROMPT_VERSION,
                          "fps": config["fps"], "groupFrames": config["groupFrames"],
                          "frameMaxSide": config["frameMaxSide"], "maxPixels": config["visionMaxPixels"],
                          "maxNewTokens": config["visionMaxNewTokens"], "quantization": config["visionQuantization"],
                          "backend": config.get("visionBackend", "transformers"),
                          "native": cache_identity(locks["vision"].get("native")),
                          "sampling": fingerprint(VISION_SAMPLING),
                          "videoUtils": versions["qwen-vl-utils"], "jsonDecoder": versions["lm-format-enforcer"]})
    return asr, vision


class Pipeline:
    def __init__(self, config: dict, locks: dict, runtime, progress):
        self.config, self.locks, self.runtime, self.progress = config, locks, runtime, progress

    def analyze(self, video: dict) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video["id"]) or video["url"] != "https://www.youtube.com/watch?v=" + video["id"]:
            raise ValueError("Identifiant ou URL YouTube invalide.")
        started = time.perf_counter()
        event("pipeline.start", "Démarrage du pipeline audio et vidéo.", videoId=video["id"])
        directory = within_project(Path(self.config["cacheDir"]) / video["id"])
        directory.mkdir(parents=True, exist_ok=True)
        # Title and channel bias recognition of names; they are part of the transcript cache key.
        context = asr_context(video, self.config)
        source_info = read_json(directory / "source.json")
        if source_info:
            original = within_project(directory / "media" / source_info.get("filename", "source.mkv"))
            source_unchanged = not original.exists() or file_sha256(original) == source_info["sha256"]
            _, key = stage_keys(source_info["sha256"], self.config, self.locks, context)
            ready = read_json(directory / "results" / key / "analysis.json")
            if source_unchanged and ready and ready.get("cacheKey") == key:
                event("pipeline.cache_hit", "Analyse complète retrouvée dans le cache ; aucune inférence nécessaire.",
                      audioChunks=ready["audioChunks"], visionGroups=ready["usage"]["visionGroups"])
                self.progress("vision", len(ready["visualSegments"]), len(ready["visualSegments"]), "Analyse retrouvée dans le cache.")
                return {**ready, "usage": {**ready["usage"], "elapsedMs": round((time.perf_counter() - started) * 1000),
                                          "cacheHits": ready["usage"]["visionGroups"] + ready["audioChunks"],
                                          "promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}}
        self.progress("download", 0, 1, "Téléchargement du média YouTube.")
        with stage("media.download", "Téléchargement du média YouTube."):
            source, receipt, warnings = media.download(video, directory / "media", self.config)
        self.progress("extract", 0, 1, "Extraction de l’audio et des captures à 1 fps.")
        with stage("media.extract", "Extraction audio PCM et captures à une image par seconde."):
            manifest = media.extract(source, directory / "media", self.config, receipt["sha256"])
        event("media.prepared", "Média prêt pour les modèles locaux.", durationMs=manifest["durationMs"],
              frameCount=len(manifest["frames"]), sourceBytes=receipt.get("sizeBytes"))
        for warning in warnings:
            event("media.warning", warning, level="warn")
        source_info = {**receipt, "durationMs": manifest["durationMs"], "sourceStartSeconds": manifest["sourceStartSeconds"]}
        atomic_json(directory / "source.json", source_info)
        asr_key, vision_key = stage_keys(receipt["sha256"], self.config, self.locks, context)
        windows = media.audio_windows(manifest["durationMs"], self.config["audioChunkSeconds"], self.config["audioOverlapSeconds"])
        hits = input_tokens = output_tokens = 0
        transcript, precise_words = [], []
        for position, window in enumerate(windows):
            chunk_started = time.perf_counter()
            chunk_fields = {"phase": "audio", "chunk": position + 1, "totalChunks": len(windows),
                            "startMs": window["startMs"], "endMs": window["endMs"]}
            event("audio.chunk.start", "Transcription du morceau audio.", **chunk_fields)
            cache = directory / "asr" / asr_key / f"{position:05}.json"
            record = read_json(cache)
            audio = directory / "media" / "chunks" / f"{position:05}.wav"
            self.progress("audio", position, len(windows), "Transcription Qwen3-ASR et alignement des mots.")
            if record and record.get("cacheKey") == asr_key and record.get("window") == window:
                result = record["result"]
                # Parse cache hits from the raw result; a saved presentation can never rewrite ASR wording.
                entries, words = utterances(result, window)
                hits += 1
                event("audio.chunk.cache_hit", "Transcription du morceau retrouvée dans le cache.", **chunk_fields)
            else:
                media.slice_wave(directory / "media" / manifest["audio"], audio, window["startMs"], window["endMs"])
                for attempt in range(2):
                    try:
                        with log_context(**chunk_fields):
                            result = self.runtime.transcribe(audio, attempt=attempt, context=context)
                        entries, words = utterances(result, window)
                        break
                    except (ValueError, KeyError) as error:
                        event("audio.chunk.retry" if not attempt else "audio.chunk.failed", str(error),
                              level="warn" if not attempt else "error", attempt=attempt + 1, **chunk_fields)
                        if attempt:
                            raise
                        self.progress("audio", position, len(windows), "Transcription ou horodatages invalides : nouvelle tentative segment par segment.")
                atomic_json(cache, {"cacheKey": asr_key, "window": window, "result": result,
                                    "words": words, "utterances": entries})
                input_tokens += result.get("inputTokens", 0)
                output_tokens += result.get("outputTokens", 0)
            if result.get("speechDetected") and not entries:
                warnings.append(f"Parole probable sans texte reconnu dans le morceau audio {position + 1}.")
            transcript.extend(entries)
            precise_words.extend(word for word in words
                                 if window["coreStartMs"] <= (word["startMs"] + word["endMs"]) / 2 < window["coreEndMs"])
            event("audio.chunk.complete", "Morceau audio transcrit et sauvegardé.", utterances=len(entries), words=len(words),
                  durationMs=round((time.perf_counter() - chunk_started) * 1000), **chunk_fields)
        transcript.sort(key=lambda entry: (entry["startMs"], entry["endMs"], entry["chunkIndex"]))
        event("audio.complete", "Transcription audio terminée.", chunks=len(windows), utterances=len(transcript), words=len(precise_words))
        frames = manifest["frames"]
        groups = [frames[start:start + self.config["groupFrames"]] for start in range(0, len(frames), self.config["groupFrames"])]
        visuals = []
        for position, group in enumerate(groups):
            group_started = time.perf_counter()
            group_fields = {"phase": "vision", "chunk": position + 1, "totalChunks": len(groups),
                            "startMs": group[0]["startMs"], "endMs": group[-1]["endMs"], "frameCount": len(group)}
            event("vision.group.start", "Analyse de la séquence vidéo.", **group_fields)
            self.progress("vision", position, len(groups), "Analyse vidéo locale avec le contexte audio.")
            cache = directory / "vision" / vision_key / f"{position:05}.json"
            record = read_json(cache)
            if record and record.get("cacheKey") == vision_key:
                observation = validate_visual(record["result"]["observation"])
                hits += 1
                event("vision.group.cache_hit", "Observation de la séquence retrouvée dans le cache.", **group_fields)
            else:
                paths = [within_project(directory / "media" / frame["path"]) for frame in group]
                context = audio_context(transcript, group[0]["startMs"], group[-1]["endMs"])
                # Malformed/unfinished JSON never counts as a completed group.
                for attempt in range(2):
                    try:
                        with log_context(**group_fields):
                            result = self.runtime.observe(paths, group, context, len(frames), attempt=attempt)
                        observation = validate_visual(result["observation"])
                        break
                    except VISION_RETRYABLE as error:
                        event("vision.group.retry" if not attempt else "vision.group.failed", str(error),
                              level="warn" if not attempt else "error", attempt=attempt + 1, **group_fields)
                        if attempt:
                            raise
                        self.progress("vision", position, len(groups), "Réponse invalide ou moteur sans réponse : nouvelle tentative avec description plus courte.")
                atomic_json(cache, {"cacheKey": vision_key, "frames": group, "audioContext": context, "result": result})
                input_tokens += result.get("inputTokens", 0)
                output_tokens += result.get("outputTokens", 0)
            visuals.append({"timestamp": timestamp(group[0]["startMs"]), "endTimestamp": timestamp(group[-1]["endMs"], round_end=True),
                            **observation, "chunkIndex": position + 1})
            event("vision.group.complete", "Observation de la séquence validée et sauvegardée.",
                  durationMs=round((time.perf_counter() - group_started) * 1000), **group_fields)
        model_label = " + ".join(self.config[key] for key in ("asrModel", "visionModel"))
        result = {
            "cacheKey": vision_key, "durationSeconds": manifest["durationSeconds"], "durationMs": manifest["durationMs"],
            "model": model_label, "promptVersion": PROMPT_VERSION, "chunkSeconds": self.config["groupFrames"],
            "transcript": [{"speaker": entry["speaker"], "timestamp": timestamp(entry["startMs"]),
                            # Rounded up like visual ends, so a 5.2-5.8 s utterance is not 00:05-00:05.
                            "endTimestamp": timestamp(min(entry["endMs"], manifest["durationMs"]), round_end=True), "text": entry["text"], "chunkIndex": entry["chunkIndex"]}
                           for entry in transcript],
            "visualSegments": visuals, "warnings": list(dict.fromkeys(warnings)), "preciseTranscript": transcript,
            "audioChunks": len(windows),
            "localProvenance": {"pipelineVersion": PIPELINE_VERSION, "models": self.locks,
                "runtimeVersions": packages(), "source": source_info, "transcriptionSource": "qwen3_asr_forced_alignment",
                "asrLanguage": self.config["asrLanguage"], "asrDecoder": ASR_DECODER,
                "asrAlignerModel": self.config["asrAlignerModel"], "asrContext": self.config.get("asrContext", "title"),
                "timestampUnits": "milliseconds", "timestampResolutionMs": 80,
                "timestampMethod": "qwen3_forced_aligner_word_timestamps",
                "speakerAttribution": "not_performed", "publicTimestampUnits": "integer_seconds",
                "visionInput": "native_video", "fps": self.config["fps"], "groupFrames": self.config["groupFrames"],
                "promptVersion": PROMPT_VERSION,
                "visionBackend": self.config.get("visionBackend", "transformers"),
                "quantization": "q4_k_m" if self.config.get("visionBackend") == "llama_cpp" else self.config["visionQuantization"],
                "words": precise_words, "utterances": transcript, "frames": frames},
            "usage": {"model": model_label, "promptTokenCount": input_tokens, "candidatesTokenCount": output_tokens,
                      "totalTokenCount": input_tokens + output_tokens, "elapsedMs": round((time.perf_counter() - started) * 1000),
                      "cacheHits": hits, "visionGroups": len(groups)},
        }
        atomic_json(directory / "results" / vision_key / "analysis.json", result)
        atomic_json(directory / "manifest.json", {"pipelineVersion": PIPELINE_VERSION, "videoId": video["id"],
                                                   "asrKey": asr_key, "visionKey": vision_key,
                                                   "audioChunks": len(windows), "videoGroups": len(groups), "status": "analyzed"})
        self.progress("vision", len(groups), len(groups), "Tous les groupes vidéo sont analysés.")
        event("pipeline.complete", "Analyse terminée ; remise des résultats au serveur pour export.",
              durationMs=result["usage"]["elapsedMs"], cacheHits=hits, audioChunks=len(windows), visionGroups=len(groups))
        return result


def cleanup(config: dict, video_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Identifiant vidéo invalide.")
    directory = within_project(Path(config["cacheDir"]) / video_id)
    manifest = read_json(directory / "manifest.json")
    if not manifest or manifest.get("status") not in ("analyzed", "exported"):
        raise ValueError("Aucune analyse complète à nettoyer.")
    if not config["keepMedia"]:
        event("media.cleanup.start", "Suppression des médias temporaires après export réussi.", videoId=video_id)
        target = within_project(directory / "media")
        if target.parent != directory:
            raise ValueError("Chemin de nettoyage invalide.")
        if target.exists():
            shutil.rmtree(target)
    atomic_json(directory / "manifest.json", {**manifest, "status": "exported"})
    event("media.cleanup.complete", "Médias temporaires nettoyés." if not config["keepMedia"] else "Médias conservés (KEEP_MEDIA=true).",
          videoId=video_id, keepMedia=config["keepMedia"])
    return {"cleaned": not config["keepMedia"]}
