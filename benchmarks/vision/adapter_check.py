"""Integration check of the actual V2 adapter, still isolated from production runs."""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "worker"))
from common import read_json, atomic_json
from main import local_environment
from runtime import LocalRuntime, model_locks
from perf import configuration

config = {**configuration(), "visionBackend": "llama_cpp", "nativeVisionDir": str(ROOT / "models/native-vision"),
          "cacheDir": str(ROOT / ".bench-vision/adapter-tmp"), "ffmpegBinary": "ffmpeg"}
local_environment(config)
runtime = LocalRuntime(config, model_locks(config))
results = []
try:
    for group, single in [(0, False), (2, False), (2, True)]:
        fixture = read_json(ROOT / ".bench-vision/fixtures" / str(group) / "fixture.json")
        paths, frames = fixture["paths"], fixture["frames"]
        if single:
            paths, frames = paths[-1:], frames[-1:]
        started = time.perf_counter()
        result = runtime.observe([Path(p) for p in paths], frames, fixture["context"], fixture["totalFrames"])
        results.append({"group": group, "frames": len(frames), "elapsedSeconds": round(time.perf_counter() - started, 3), **result})
    native_pid = runtime.native.child.pid
    runtime.load("asr")
    assert runtime.native is None
    assert runtime.role == "asr"
    results.append({"switchToAsr": True, "nativeProcessClosed": True, "oldNativePid": native_pid})
    atomic_json(ROOT / ".bench-vision/results/adapter-check.json", results)
    print(json.dumps([{k: v for k, v in r.items() if k != "observation"} for r in results]))
finally:
    runtime.unload()
