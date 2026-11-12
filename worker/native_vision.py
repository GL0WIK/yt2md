"""Managed offline llama.cpp CUDA backend for Qwen3.5 video observations."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from common import PROJECT_ROOT, read_json, within_project
from eventlog import event, stage

# Part of the vision cache key: any change to how the engine decodes recomputes observations.
SERVER_ARGS = ["-ngl", "99", "-fa", "on", "-c", "8192", "-np", "1", "--video-fps", "1",
               "--video-timestamp-interval", "1000", "--reasoning", "off", "--jinja"]
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.1}
PATH_FIELDS = ("binary", "model", "projector")
# Scratch folders of a crashed worker are older than this process.
PROCESS_STARTED = time.time()


class TransientVisionError(RuntimeError):
    """Engine timeout or refusal: the group is retried (after an engine restart when needed)."""


def cache_identity(lock):
    # Cache keys must not depend on where the project is checked out.
    if not lock:
        return lock
    paths = {name: Path(lock[name]).resolve().relative_to(PROJECT_ROOT).as_posix()
             for name in PATH_FIELDS if lock.get(name)}
    return {**{key: value for key, value in lock.items() if key not in PATH_FIELDS}, **paths,
            "serverArgs": SERVER_ARGS, "sampling": SAMPLING}


def bind_to_worker(child):
    """Windows: kill llama-server with this process, even on a hard crash. Best-effort, no-op elsewhere."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not (kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
                and kernel32.AssignProcessToJobObject(job, int(child._handle))):
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def release_job(job):
    if job:
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle(job)
        except Exception:
            pass


def remove_stale_scratch(cache: Path):
    """Remove scratch folders left by a crashed worker; folders still in use are kept."""
    for folder in cache.glob("native-vision-*"):
        try:
            if not folder.is_dir():
                continue
            if not folder.name.startswith("native-vision-stale-"):
                if max([folder.stat().st_mtime, *(item.stat().st_mtime for item in folder.iterdir())]) > PROCESS_STARTED - 60:
                    continue
                # Renaming fails on Windows while a live engine still holds a file open.
                renamed = folder.with_name("native-vision-stale-" + secrets.token_hex(4))
                folder.rename(renamed)
                folder = renamed
            target = within_project(folder)
            if target.parent == cache:
                shutil.rmtree(target)
                event("native.scratch.cleanup", "Dossier temporaire orphelin du moteur natif supprimé.", level="debug")
        except OSError:
            continue


def native_lock(config):
    directory = within_project(config["nativeVisionDir"])
    lock = read_json(directory / "lock.json")
    if not lock or lock.get("modelRepo") != "unsloth/Qwen3.5-9B-GGUF" or not lock.get("modelRevision"):
        raise RuntimeError("Moteur vidéo natif absent. Exécuter npm run models:download -- vision.")
    if config["visionModel"] != "Qwen/Qwen3.5-9B":
        raise RuntimeError("Le moteur natif validé utilise Qwen/Qwen3.5-9B.")
    paths = {"binary": directory / "bin/llama-server.exe",
             "model": directory / "models/Qwen3.5-9B-Q4_K_M.gguf",
             "projector": directory / "models/mmproj-F16.gguf"}
    for name, path in paths.items():
        path = within_project(path)
        if not path.is_relative_to(directory) or not path.is_file():
            raise RuntimeError(f"Fichier vidéo natif {name} absent. Relancer models:download -- vision.")
        paths[name] = str(path)
    return {**lock, **paths, "backend": "llama_cpp", "quantization": "q4_k_m"}


def request_payload(encoded_video, frames, context, schema, limit):
    from runtime import SYSTEM_PROMPT
    metadata = {"interval": {"startMs": frames[0]["startMs"], "endMs": frames[-1]["endMs"]},
                "observedFrameTimesMs": [frame["ptsMs"] for frame in frames],
                "audioContext": context, "outputSchema": schema}
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "input_video", "input_video": {"url": "data:video/mp4;base64," + encoded_video}},
            {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "visual", "schema": schema, "strict": True}},
        "max_tokens": limit, **SAMPLING, "cache_prompt": False, "timings": True}


class NativeVision:
    def __init__(self, config, lock):
        self.config, self.lock = config, lock
        self.child = self.log = self.job = None
        self.key = secrets.token_urlsafe(24)
        self.url = None
        cache = within_project(config["cacheDir"])
        cache.mkdir(parents=True, exist_ok=True)
        remove_stale_scratch(cache)
        self.scratch = within_project(Path(tempfile.mkdtemp(prefix="native-vision-", dir=cache)))
        self.previous_clip = None

    def start(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log = (self.scratch / "engine.log").open("w", encoding="utf-8")
        # The key never appears on a command line readable by other local processes.
        key_file = self.scratch / "api-key.txt"
        key_file.write_text(self.key + "\n", encoding="utf-8")
        command = [self.lock["binary"], "-m", self.lock["model"], "--mmproj", self.lock["projector"],
                   "--host", "127.0.0.1", "--port", str(port), "--api-key-file", str(key_file),
                   "--device", "CUDA" + self.config["device"].split(":")[-1], *SERVER_ARGS,
                   "--image-max-tokens", str(self.config["visionMaxPixels"] // 1024), "--perf", "-lv", "0"]
        try:
            self.child = subprocess.Popen(command, stdout=self.log, stderr=self.log,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.job = bind_to_worker(self.child)
            started = time.perf_counter()
            while True:
                if self.child.poll() is not None:
                    raise RuntimeError("Le moteur CUDA natif s’est arrêté pendant son chargement.")
                try:
                    self.request("/health", timeout=1)
                    break
                except (urllib.error.URLError, TimeoutError) as error:
                    if isinstance(error, urllib.error.HTTPError) and error.code != 503:
                        raise
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("Chargement du moteur CUDA natif trop long.")
                    time.sleep(0.2)
            key_file.unlink(missing_ok=True)
            event("native.started", "Moteur vidéo CUDA natif prêt, raisonnement désactivé.",
                  pid=self.child.pid, engine=self.lock["engine"], quantization="q4_k_m")
        except Exception:
            # The scratch folder stays usable for a later restart; close() removes it.
            key_file.unlink(missing_ok=True)
            self.stop()
            raise

    def request(self, path, payload=None, timeout=180):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.url + path, body,
                    {"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if path == "/health":
                raise
            try:
                message = json.loads(error.read()).get("error", {}).get("message", "Requête refusée.")
            except (ValueError, AttributeError):
                message = "Requête refusée."
            raise TransientVisionError(f"Moteur vidéo natif HTTP {error.code} : {str(message)[:1000]}") from error

    def encode(self, paths, frames):
        paths = [within_project(path) for path in paths]
        signature = tuple((str(path), frame["startMs"], frame["endMs"]) for path, frame in zip(paths, frames))
        clip = self.scratch / "clip.mp4"
        if signature != self.previous_clip:
            if not paths or len(paths) != len(frames):
                raise ValueError("Captures vidéo absentes ou incohérentes.")
            listing = self.scratch / "frames.ffconcat"
            lines = ["ffconcat version 1.0"]
            for path in paths:
                escaped = path.as_posix().replace("'", "'\\''")
                lines += [f"file '{escaped}'", "duration 1"]
            listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
            from media import run
            with stage("vision.clip", "Préparation de la vidéo native à une image par seconde.", inputFrames=len(frames)):
                run(self.config["ffmpegBinary"], ["-hide_banner", "-loglevel", "error", "-y",
                     "-f", "concat", "-safe", "0", "-i", str(listing), "-r", "1",
                     "-frames:v", str(len(frames)), "-c:v", "libx264rgb", "-crf", "0",
                     "-preset", "ultrafast", "-pix_fmt", "rgb24", str(clip)], 60, operation="media.native_video")
            self.previous_clip = signature
        return base64.b64encode(clip.read_bytes()).decode("ascii")

    def observe(self, paths, frames, context, attempt=0):
        from runtime import generation_schema, parse_visual_output
        encoded = self.encode(paths, frames)
        limit = min(8192, self.config["visionMaxNewTokens"] * (attempt + 1))
        if self.child is None or self.child.poll() is not None:
            event("native.restart", "Moteur vidéo natif arrêté : redémarrage.", level="warn")
            self.restart()
        with stage("vision.generate", "Analyse vidéo Qwen sur CUDA natif et génération du JSON.",
                   inputFrames=len(frames), maxNewTokens=limit, attempt=attempt + 1, backend="llama_cpp"):
            try:
                response = self.request("/v1/chat/completions", request_payload(encoded, frames, context, generation_schema(attempt), limit))
            except (TimeoutError, urllib.error.URLError) as error:
                # With -np 1 the engine keeps decoding an abandoned request and would block the next one.
                event("native.restart", "Requête native sans réponse : redémarrage du moteur.", level="warn",
                      errorType=type(error).__name__)
                self.restart()
                raise TransientVisionError("Moteur vidéo natif sans réponse : " + str(error)) from error
        usage = response["usage"]
        raw = response["choices"][0]["message"].get("content")
        if not isinstance(raw, str):
            raise ValueError("Moteur vidéo natif : réponse sans contenu texte.")
        value = parse_visual_output(raw, usage["completion_tokens"], limit)
        timings = response.get("timings", {})
        event("vision.generate.result", "Observation JSON native obtenue et validée.",
              outputTokens=usage["completion_tokens"], inputTokens=usage["prompt_tokens"],
              tokensPerSecond=round(timings.get("predicted_per_second", 0), 2),
              nativePromptMs=round(timings.get("prompt_ms", 0)), backend="llama_cpp",
              screenTextEntries=len(value["screenText"]), uncertainties=len(value["uncertainties"]))
        return {"observation": value, "inputTokens": usage["prompt_tokens"], "outputTokens": usage["completion_tokens"],
                "backend": "llama_cpp", "engine": self.lock["engine"], "quantization": "q4_k_m", "timings": timings}

    def restart(self):
        self.stop()
        self.start()

    def stop(self):
        if self.child is not None:
            if self.child.poll() is None:
                self.child.terminate()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
            self.child = None
        release_job(self.job)
        self.job = None
        if self.log:
            self.log.close()
            self.log = None

    def close(self):
        self.stop()
        target = within_project(self.scratch)
        cache = within_project(self.config["cacheDir"])
        if target != cache and target.is_relative_to(cache) and target.exists():
            shutil.rmtree(target)
        self.previous_clip = None
