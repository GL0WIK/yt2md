"""Download a pinned native CUDA engine and Qwen9B quantization only into the bench."""
from pathlib import Path
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".bench-vision/llama"
BENCH.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = str(BENCH / "hf-cache")
from huggingface_hub import hf_hub_download, HfApi

tag = "b11576"
revision = "3885219b6810b007914f3a7950a8d1b469d598a5"
info = json.load(urllib.request.urlopen(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"))
files = [f"llama-{tag}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
records = []
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for name in files:
    asset = next(asset for asset in info["assets"] if asset["name"] == name)
    destination = BENCH / name
    if not destination.exists():
        print("Downloading", name, flush=True)
        urllib.request.urlretrieve(asset["browser_download_url"], destination)
    digest = sha256(destination)
    if asset.get("digest") and asset["digest"] != "sha256:" + digest:
        raise RuntimeError("Release asset checksum mismatch")
    folder = BENCH / "bin"
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination) as archive:
        for member in archive.infolist():
            if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                raise RuntimeError("Zip path outside benchmark")
        archive.extractall(folder)
    records.append({"file": name, "sha256": digest, "url": asset["browser_download_url"]})
for filename in ("Qwen3.5-9B-Q4_K_M.gguf", "mmproj-F16.gguf"):
    print("Downloading", filename, flush=True)
    destination = Path(hf_hub_download("unsloth/Qwen3.5-9B-GGUF", filename, revision=revision,
                       local_dir=BENCH / "models"))
    records.append({"file": filename, "bytes": destination.stat().st_size,
                    "sha256": sha256(destination)})
(BENCH / "lock.json").write_text(json.dumps({"engine": tag, "modelRevision": revision,
                    "modelRepo": "unsloth/Qwen3.5-9B-GGUF", "files": records}, indent=2), encoding="utf-8")
print("Native CUDA benchmark ready", flush=True)
