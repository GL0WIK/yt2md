"""Small dependency-free helpers. Runtime writes are confined to this project."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "qwen3-asr-native-video-v4"
PROMPT_VERSION = "local-observed-video-v2-compact"


def within_project(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path == PROJECT_ROOT or not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Le chemin doit rester dans le dossier du projet : {value}")
    return path


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = within_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Cache JSON trop volumineux.")
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def timestamp(milliseconds: int, *, round_end: bool = False) -> str:
    seconds = math.ceil(milliseconds / 1000) if round_end else milliseconds // 1000
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"
