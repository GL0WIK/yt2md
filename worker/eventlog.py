"""Small structured journal; stdout remains exclusively the worker protocol."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime, timezone
import re
import sys
import threading
import time

_context = ContextVar("local_log_context", default=({}, None))


def clean(value: str) -> str:
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = re.sub(r"\b(?:hf_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,})\b", "[redacted]", value)
    value = re.sub(r"(?i)\b(authorization|api[_-]?key|access[_-]?token|password|cookie|secret)\s*[:=]\s*[^\s,;]+",
                   r"\1=[redacted]", value)
    return value[:4000]


@contextmanager
def log_context(sink=None, **fields):
    previous_fields, previous_sink = _context.get()
    token = _context.set(({**previous_fields, **fields}, sink or previous_sink))
    try:
        yield
    finally:
        _context.reset(token)


def event(name: str, message: str, *, level: str = "info", **fields):
    context_fields, sink = _context.get()
    fields = {key: clean(value) if isinstance(value, str) else value
              for key, value in {**context_fields, **fields}.items()
              if value is None or isinstance(value, (str, int, float, bool))}
    message = clean(message)
    if sink:
        sink({"event": "log", "name": name, "level": level, "message": message, "fields": fields})
    else:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"[{timestamp}] {level.upper()} {name} — {message}" + (" | " + details if details else ""),
              file=sys.stderr, flush=True)


@contextmanager
def stage(name: str, message: str, *, heartbeat_seconds: float = 15, heartbeat_fields=None, **fields):
    started = time.perf_counter()
    event(name + ".start", message, **fields)
    stopped = threading.Event()
    copied_context = copy_context()

    def waiting():
        while not stopped.wait(heartbeat_seconds):
            extra = {}
            if heartbeat_fields:
                try:
                    extra = heartbeat_fields()
                except Exception:
                    pass  # Optional monitoring must never stop inference.
            if stopped.is_set():
                return
            event(name + ".waiting", message + " — traitement toujours actif.",
                  elapsedMs=round((time.perf_counter() - started) * 1000), **fields, **extra)

    thread = threading.Thread(target=lambda: copied_context.run(waiting), daemon=True)
    thread.start()
    try:
        yield
    except Exception as error:
        stopped.set()
        thread.join(timeout=1)
        extra = {}
        if heartbeat_fields:
            try:
                extra = heartbeat_fields()
            except Exception:
                pass
        event(name + ".failed", str(error), level="error", errorType=type(error).__name__,
              durationMs=round((time.perf_counter() - started) * 1000), **fields, **extra)
        raise
    else:
        stopped.set()
        thread.join(timeout=1)
        event(name + ".complete", message, durationMs=round((time.perf_counter() - started) * 1000), **fields)
    finally:
        stopped.set()
        thread.join(timeout=1)
