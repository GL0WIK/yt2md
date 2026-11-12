"""Native CUDA video benchmark on a dedicated localhost port. Production stays untouched."""
from pathlib import Path
import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".bench-vision"
sys.path.insert(0, str(ROOT / "worker"))
from common import read_json, atomic_json
from runtime import SYSTEM_PROMPT, generation_schema, parse_visual_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    fixture = read_json(BENCH / "fixtures" / str(args.group) / "fixture.json")
    folder = BENCH / "fixtures" / str(args.group)
    clip = folder / "clip.mp4"
    # lossless RGB encoding of the copied JPEG pixels, exactly one frame/second.
    first = fixture["frames"][0]["index"]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1",
                    "-start_number", str(first), "-i", str(folder / "frame-%06d.jpg"),
                    "-frames:v", str(len(fixture["frames"])), "-c:v", "libx264rgb", "-crf", "0",
                    "-preset", "ultrafast", "-pix_fmt", "rgb24", str(clip)], check=True)
    exe = BENCH / "llama/bin/llama-server.exe"
    model_dir = BENCH / "llama/models"
    command = [str(exe), "-m", str(model_dir / "Qwen3.5-9B-Q4_K_M.gguf"),
               "--mmproj", str(model_dir / "mmproj-F16.gguf"), "--host", "127.0.0.1", "--port", "18083",
               "-ngl", "99", "-fa", "on", "-c", "8192", "-np", "1", "--video-fps", "1",
               "--video-timestamp-interval", "1000", "--image-max-tokens", "384", "--reasoning", "off",
               "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}', "--perf"]
    log_path = BENCH / "results/llama-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            started = time.perf_counter()
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Native server exited ({child.returncode}), see {log_path}")
                try:
                    urllib.request.urlopen("http://127.0.0.1:18083/health", timeout=1).close()
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("Native model loading timed out")
                    time.sleep(0.5)
            print("Native CUDA model ready", flush=True)
            schema = generation_schema()
            context = {"interval": {"startMs": fixture["frames"][0]["startMs"], "endMs": fixture["frames"][-1]["endMs"]},
                       "observedFrameTimesMs": [frame["ptsMs"] for frame in fixture["frames"]],
                       "audioContext": fixture["context"], "outputSchema": schema}
            for repeat in range(args.runs):
                body = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "input_video", "input_video": {"url": "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()}},
                        {"type": "text", "text": json.dumps(context, ensure_ascii=False)}]}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "visual", "schema": schema, "strict": True}},
                    "max_tokens": 1024, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.1,
                    "seed": 42 + repeat, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False,
                    "timings": True}
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                request = urllib.request.Request("http://127.0.0.1:18083/v1/chat/completions", payload, {"Content-Type": "application/json"})
                start = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        result = json.load(response)
                except urllib.error.HTTPError as error:
                    raise RuntimeError(error.read().decode("utf-8")) from error
                elapsed = time.perf_counter() - start
                raw = result["choices"][0]["message"]["content"]
                observation = parse_visual_output(raw, result["usage"]["completion_tokens"], 1024)
                receipt = {"profile": "llama-cuda-q4km", "group": args.group, "repeat": repeat,
                           "fixtureHashes": fixture["sha256"], "nativeVideo": True, "frames": len(fixture["frames"]),
                           "wallSeconds": round(elapsed, 3), "validJson": True, "response": result, "observation": observation}
                atomic_json(BENCH / "results" / f"llama-video-g{args.group}-r{repeat}.json", receipt)
                print(json.dumps({k: v for k, v in receipt.items() if k not in ("fixtureHashes", "response", "observation")}
                      | {"timings": result.get("timings"), "usage": result["usage"]}), flush=True)
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
