"""JSON-lines worker; stdout is reserved for protocol, ML messages go to stderr."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback

from common import within_project
from eventlog import event, log_context


def local_environment(config: dict):
    models = within_project(config["modelsDir"])
    os.environ["HF_HOME"] = str(models / ".hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # torch.compile kernels persist across worker launches, inside the project like every other cache.
    compiled = within_project(Path(config["cacheDir"]) / "torch-compile") if config.get("cacheDir") else None
    if compiled:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(compiled / "inductor"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(compiled / "triton"))
    venv_bin = str(Path(sys.executable).parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    # Called on every request: prepend once instead of growing PATH without bound.
    if os.path.normcase(os.path.normpath(venv_bin)) not in {os.path.normcase(os.path.normpath(e)) for e in entries if e}:
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


def protocol_output():
    """Keep a private copy of the real stdout for the protocol; fd 1 then points to stderr.

    contextlib.redirect_stdout only covers Python writes: native libraries (CUDA, ONNX, llama.cpp
    bindings) writing to fd 1 would otherwise corrupt the JSON lines read by the server.
    """
    sys.stdout.flush()
    stream = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return stream


def serve(output=None):
    from pipeline import Pipeline, cleanup
    from runtime import LocalRuntime, doctor, model_locks, runtime_fingerprint
    output = output or sys.stdout
    runtime, runtime_key = None, None
    output_lock = threading.Lock()

    def emit(value):
        with output_lock:
            output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()

    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                config = request["config"]
                local_environment(config)
                action = request["action"]
                def log_sink(value, request_id=request["id"]):
                    emit({"id": request_id, **value})
                with log_context(sink=log_sink, videoId=request.get("video", {}).get("id") or request.get("videoId")), contextlib.redirect_stdout(sys.stderr):
                    if action == "health":
                        value = doctor(config)
                    elif action == "cleanup":
                        value = cleanup(config, request["videoId"])
                    elif action == "analyze":
                        event("worker.analysis.start", "Vérification des modèles et initialisation du pipeline.")
                        locks = model_locks(config)
                        key = runtime_fingerprint(config, locks)
                        if key != runtime_key:
                            if runtime:
                                runtime.unload()
                            runtime, runtime_key = LocalRuntime(config, locks), key
                        else:
                            # keepMedia, timeouts, binaries...: no reason to reload the models.
                            runtime.update_config(config)
                        def progress(phase, chunk, total, detail):
                            emit({"id": request["id"], "event": "progress", "phase": phase,
                                  "chunk": chunk, "totalChunks": total, "detail": detail})
                        value = Pipeline(config, locks, runtime, progress).analyze(request["video"])
                    elif action == "shutdown":
                        if runtime:
                            runtime.unload()
                        emit({"id": request["id"], "ok": True, "result": {"stopped": True}})
                        return
                    else:
                        raise ValueError("Action worker inconnue.")
                emit({"id": request["id"], "ok": True, "result": value})
            except Exception as error:
                # An unparseable line has no id: the reply carries id null, which the server ignores.
                with log_context(sink=lambda value, request_id=request.get("id"): emit({"id": request_id, **value})):
                    event("worker.request.failed", str(error), level="error", errorType=type(error).__name__,
                          traceback="".join(traceback.format_exception(type(error), error, error.__traceback__, limit=6)))
                emit({"id": request.get("id"), "ok": False,
                      "error": {"type": type(error).__name__, "message": str(error)}})
    finally:
        if runtime:
            runtime.unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if args.download:
        from download_models import download_models
        request = json.load(sys.stdin)
        print(json.dumps(download_models(request), ensure_ascii=False))
    elif args.stdio:
        serve(protocol_output())
    else:
        parser.error("Utiliser --stdio ou --download.")


if __name__ == "__main__":
    main()
