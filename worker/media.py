"""yt-dlp and FFmpeg media preparation, without loading inference dependencies."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from contextvars import copy_context
import threading
import time
import wave

from common import atomic_json, file_sha256, read_json, within_project
from eventlog import event, stage


def run(binary: str, args: list[str], timeout: float, *, cwd: Path | None = None, operation: str = "media.tool",
        progress_fields: dict | None = None):
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    with stage(operation, "Exécution de " + Path(binary).name + ".", binary=Path(binary).name):
        child = subprocess.Popen([binary, *args], cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, creationflags=flags)
        buffers = {"stdout": [], "stderr": []}
        overflow = threading.Event()

        def consume(pipe, stream):
            size = 0
            last_progress = 0
            with pipe:
                for raw in iter(pipe.readline, b""):
                    size += len(raw)
                    if size <= 64 * 1024 * 1024:
                        buffers[stream].append(raw)
                    else:
                        overflow.set()
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("LOCAL_PROGRESS:"):
                        values = line.removeprefix("LOCAL_PROGRESS:").split("|")
                        if len(values) == 3:
                            event("media.download.progress", "Téléchargement en cours.", percent=values[0].strip(),
                                  speed=values[1].strip(), eta=values[2].strip())
                    elif "pts_time:" in line and "showinfo" in line:
                        match = re.search(r"\bn:\s*(\d+).*?\bpts_time:([-+\d.eE]+)", line)
                        if match and time.monotonic() - last_progress >= 2:
                            last_progress = time.monotonic()
                            event("media.frames.progress", "Extraction des captures en cours.",
                                  extractedFrames=int(match[1]) + 1, mediaTimeMs=round(float(match[2]) * 1000),
                                  **(progress_fields or {}))
                    elif line.startswith("out_time_us=") and time.monotonic() - last_progress >= 2:
                        last_progress = time.monotonic()
                        value = line.split("=", 1)[1]
                        if value.isdigit():
                            event("media.audio.progress", "Extraction de l’audio en cours.", mediaTimeMs=int(value) // 1000,
                                  **(progress_fields or {}))
                    elif operation == "media.ytdlp" and line:
                        event("media.download.output", line, level="warn" if "WARNING:" in line or "ERROR:" in line else "info")

        threads = []
        for stream, pipe in [("stdout", child.stdout), ("stderr", child.stderr)]:
            copied_context = copy_context()
            thread = threading.Thread(target=copied_context.run, args=(consume, pipe, stream), daemon=True)
            thread.start()
            threads.append(thread)
        try:
            child.wait(timeout=timeout)
        except BaseException:
            # yt-dlp may own a merging FFmpeg process: stop that child too on Windows.
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                try:
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=flags, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    child.kill()
            if child.poll() is None:
                child.kill()
            child.wait()
            raise
        finally:
            for thread in threads:
                thread.join(timeout=5)
        stdout = b"".join(buffers["stdout"]).decode("utf-8", errors="replace")
        stderr = b"".join(buffers["stderr"]).decode("utf-8", errors="replace")
        if child.returncode:
            raise RuntimeError(f"{Path(binary).name} a échoué ({child.returncode}) : {stderr[-1600:]}")
        if overflow.is_set():
            raise RuntimeError("Sortie de l’outil média trop volumineuse.")
        return stdout, stderr


def audio_track_warnings(info: dict, chosen: list[dict]) -> list[str]:
    # YouTube only labels a track "original" when the video offers several (dubbed) languages.
    languages = {f.get("language") for f in info.get("formats", [])
                 if f.get("acodec") not in (None, "none") and f.get("language")}
    if len(languages) > 1 and not any("original" in str(f.get("format_note", "")).lower() for f in chosen):
        return ["La piste audio sélectionnée n’est pas explicitement marquée « original » par YouTube. Vérifier sa langue dans le cache média."]
    return []


def download(video: dict, directory: Path, config: dict) -> tuple[Path, dict, list[str]]:
    directory = within_project(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = read_json(directory / "download.json")
    if receipt:
        source = within_project(directory / receipt["filename"])
        if source.is_file() and source.stat().st_size == receipt["sizeBytes"]:
            # Recheck content on restart; an interrupted/external overwrite must invalidate caches.
            if file_sha256(source) == receipt["sha256"]:
                event("media.download.cache_hit", "Média téléchargé retrouvé et empreinte vérifiée.", sourceBytes=receipt["sizeBytes"])
                info = read_json(directory / "source.info.json") or {}
                return source, receipt, audio_track_warnings(info, receipt.get("audioFormats", []))
    stdout, _ = run(config["ytDlpBinary"], [
        "--ignore-config", "--no-cache-dir", "--no-playlist", "--no-quiet", "--progress", "--newline", "--restrict-filenames",
        "--progress-delta", "2", "--progress-template", "download:LOCAL_PROGRESS:%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
        "--force-overwrites",
        "--js-runtimes", "node:" + config["nodeBinary"],
        "--write-info-json", "--merge-output-format", "mkv",
        "--ffmpeg-location", shutil.which(config["ffmpegBinary"]) or config["ffmpegBinary"],
        *config.get("ytDlpAuthArgs", []),
        "--format", "bv[height<=1080]+ba[format_note*=original]/bv[height<=1080]+ba/b[height<=1080]/best",
        "--paths", str(directory), "--output", "source.%(ext)s", "--print", "after_move:LOCAL_FILE:%(filepath)s",
        "--", video["url"],
    ], config["downloadTimeoutMs"] / 1000, cwd=directory, operation="media.ytdlp")
    candidates = [Path(line.removeprefix("LOCAL_FILE:").strip()) for line in stdout.splitlines() if line.startswith("LOCAL_FILE:")]
    source = next((within_project(p if p.is_absolute() else directory / p)
                   for p in reversed(candidates) if (p if p.is_absolute() else directory / p).is_file()), None)
    if source is None or source.parent != directory:
        raise RuntimeError("yt-dlp n’a pas fourni de fichier média dans le dossier demandé.")
    info = read_json(directory / "source.info.json") or {}
    if info.get("id") != video["id"]:
        raise RuntimeError("Le média téléchargé ne correspond pas à l’identifiant YouTube demandé.")
    audio_formats = [f for f in info.get("requested_formats", [info]) if f.get("acodec") not in (None, "none")]
    chosen_audio = [{key: f.get(key) for key in ("format_id", "language", "format_note")} for f in audio_formats]
    warnings = audio_track_warnings(info, audio_formats)
    receipt = {"filename": source.name, "sizeBytes": source.stat().st_size,
               "sha256": file_sha256(source), "audioFormats": chosen_audio, "warnings": warnings}
    atomic_json(directory / "download.json", receipt)
    event("media.download.verified", "Téléchargement terminé ; identifiant et empreinte du média vérifiés.",
          filename=source.name, sourceBytes=receipt["sizeBytes"])
    return source, receipt, warnings


def probe(source: Path, config: dict) -> dict:
    stdout, _ = run(config["ffprobeBinary"], ["-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)], 120, operation="media.probe")
    result = json.loads(stdout)
    duration = float(result["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0 or duration > 24 * 3600:
        raise ValueError("Durée du fichier média invalide (maximum 24 heures).")
    if not any(s.get("codec_type") == "video" for s in result["streams"]):
        raise ValueError("Aucune piste vidéo dans le média.")
    if not any(s.get("codec_type") == "audio" for s in result["streams"]):
        raise ValueError("Aucune piste audio dans le média.")
    return {"durationMs": round(duration * 1000), "durationSeconds": math.ceil(duration),
            "sourceStartSeconds": float(result["format"].get("start_time", 0)), "streams": result["streams"]}


def stream_duration_ms(stream: dict) -> int | None:
    """Stream duration from ffprobe: `duration` (MP4/WebM) or the Matroska DURATION tag."""
    try:
        if stream.get("duration") not in (None, "N/A"):
            value = float(stream["duration"])
        else:
            tag = next(v for k, v in (stream.get("tags") or {}).items() if k.upper() == "DURATION")
            hours, minutes, seconds = tag.split(":")
            value = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (StopIteration, ValueError, TypeError, AttributeError):
        return None
    return round(value * 1000) if math.isfinite(value) and value > 0 else None


# A video stream may end shortly before the audio: those last seconds have no image to capture.
MAX_MISSING_TAIL_FRAMES = 2


def check_frame_coverage(count: int, details: dict) -> None:
    expected = math.ceil(details["durationMs"] / 1000)
    if count == expected:
        return
    video = next((s for s in details["streams"] if s.get("codec_type") == "video"), {})
    video_ms = stream_duration_ms(video)
    # Accepted only when every second of the video stream has its frame (100 ms timestamp tolerance).
    if (count < expected and expected - count <= MAX_MISSING_TAIL_FRAMES and video_ms is not None
            and video_ms < details["durationMs"] and count * 1000 + 100 >= video_ms):
        event("media.frames.short_video", "La piste vidéo se termine avant l’audio ; dernières secondes sans capture.",
              level="warn", frameCount=count, expectedFrames=expected, videoDurationMs=video_ms,
              durationMs=details["durationMs"])
        return
    raise RuntimeError("FFmpeg : certaines secondes ne disposent pas de capture vidéo.")


def extract(source: Path, directory: Path, config: dict, source_hash: str) -> dict:
    directory = within_project(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "frames.json"
    cached = read_json(manifest_path)
    audio = directory / "audio.wav"
    if cached and cached.get("sourceHash") == source_hash and cached.get("frameMaxSide") == config["frameMaxSide"]:
        if audio.is_file() and all((directory / f["path"]).is_file() for f in cached["frames"]):
            event("media.extract.cache_hit", "Audio et captures retrouvés dans le cache.", frameCount=len(cached["frames"]))
            return cached
    details = probe(source, config)
    # Lets the journal show extraction as a share of the video.
    progress = {"mediaDurationMs": details["durationMs"]}
    # An interrupted extraction must never leave a truncated audio.wav that a later cache hit would accept.
    partial = directory / "audio.partial.wav"
    try:
        run(config["ffmpegBinary"], ["-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-y", "-i", str(source),
            "-map", "0:a:0", "-vn", "-af", "aresample=16000:async=1:first_pts=0", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(partial)], config["downloadTimeoutMs"] / 1000, operation="media.audio.extract",
            progress_fields=progress)
        os.replace(partial, audio)
        # Progress is sampled every 2 s: a short extraction must still end its bar at 100 %.
        event("media.audio.progress", "Audio extrait.", mediaTimeMs=details["durationMs"], **progress)
    finally:
        partial.unlink(missing_ok=True)
    frames_dir = within_project(directory / "frames")
    frames_dir.mkdir(exist_ok=True)
    # The directory is owned by this extraction stage. Remove only its numbered JPEGs.
    for old in frames_dir.glob("frame-*.jpg"):
        within_project(old).unlink()
    side = config["frameMaxSide"]
    try:
        stderr = extract_frames(source, frames_dir, side, config, progress, gpu=True)
    except RuntimeError as error:
        # Rare codecs, or no CUDA decoder: the processor path gives the same frames, more slowly.
        event("media.frames.cpu_fallback", "Décodage GPU impossible pour ce média ; images extraites par le processeur.",
              level="warn", reason=str(error)[-400:])
        for old in frames_dir.glob("frame-*.jpg"):
            within_project(old).unlink()
        stderr = extract_frames(source, frames_dir, side, config, progress, gpu=False)
    event("media.frames.progress", "Captures extraites.", mediaTimeMs=details["durationMs"], **progress)
    pts = re.findall(r"\bn:\s*(\d+)\s+pts:.*?\bpts_time:([-+\d.eE]+)", stderr)
    paths = sorted(frames_dir.glob("frame-*.jpg"))
    if not paths or len(paths) != len(pts):
        raise RuntimeError("FFmpeg : nombre de captures et horodatages incohérent.")
    frames = []
    for index, (path, (number, time)) in enumerate(zip(paths, pts)):
        if int(number) != index:
            raise RuntimeError("FFmpeg : ordre des captures incohérent.")
        start = index * 1000
        if start >= details["durationMs"]:
            continue
        frames.append({"index": index, "path": str(path.relative_to(directory)).replace("\\", "/"),
                       "ptsMs": round(float(time) * 1000), "startMs": start,
                       "endMs": min(start + 1000, details["durationMs"])})
    check_frame_coverage(len(frames), details)
    result = {**details, "sourceHash": source_hash, "frameMaxSide": side, "fps": 1, "audio": "audio.wav", "frames": frames}
    atomic_json(manifest_path, result)
    return result


def extract_frames(source: Path, frames_dir: Path, side: int, config: dict, progress: dict, *, gpu: bool) -> str:
    """One JPEG per second. On the GPU, NVDEC decodes and scale_cuda resizes: about 5 times faster
    than CPU decoding of a 1080p AV1 video, which only keeps 1 frame out of 25."""
    timing = "setpts=PTS-STARTPTS,fps=1:start_time=0:round=down:eof_action=pass"
    fit = f"w='min({side},iw)':h='min({side},ih)':force_original_aspect_ratio=decrease"
    if gpu:
        device = config.get("device", "cuda:0").split(":")[-1]
        decode = ["-hwaccel", "cuda", "-hwaccel_device", device, "-hwaccel_output_format", "cuda"]
        # format=yuv420p also converts 10-bit sources before the download to system memory.
        chain = f"{timing},scale_cuda={fit}:interp_algo=lanczos:format=yuv420p,hwdownload,format=yuv420p,showinfo"
    else:
        decode = []
        chain = f"{timing},scale={fit}:flags=lanczos,showinfo"
    _, stderr = run(config["ffmpegBinary"], ["-hide_banner", "-loglevel", "info", "-nostats", "-y", *decode, "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", chain, "-fps_mode", "vfr", "-pix_fmt", "yuvj420p", "-q:v", "2",
        "-start_number", "0", str(frames_dir / "frame-%06d.jpg")],
        config["downloadTimeoutMs"] / 1000, operation="media.frames.extract", progress_fields={**progress, "decoder": "nvdec" if gpu else "cpu"})
    return stderr


def audio_windows(duration_ms: int, seconds: int, overlap_seconds: int) -> list[dict]:
    windows = []
    width, overlap = seconds * 1000, overlap_seconds * 1000
    for index, start in enumerate(range(0, duration_ms, width)):
        end = min(start + width, duration_ms)
        windows.append({"index": index, "coreStartMs": start, "coreEndMs": end,
                        "startMs": max(0, start - overlap), "endMs": min(duration_ms, end + overlap)})
    return windows


def slice_wave(source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    target = within_project(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as incoming:
        if incoming.getframerate() != 16000 or incoming.getnchannels() != 1 or incoming.getsampwidth() != 2:
            raise ValueError("Audio PCM 16 kHz mono attendu.")
        offset = min(incoming.getnframes(), start_ms * 16)
        incoming.setpos(offset)
        data = incoming.readframes(max(0, end_ms * 16 - offset))
    with wave.open(str(target), "wb") as outgoing:
        outgoing.setnchannels(1)
        outgoing.setsampwidth(2)
        outgoing.setframerate(16000)
        outgoing.writeframes(data)
