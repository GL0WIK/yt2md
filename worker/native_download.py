"""Pinned, verified native Metal (Apple Silicon) installation. Network is used only on explicit download."""
from pathlib import Path
import os
import shutil
import tarfile
import urllib.request

from common import PROJECT_ROOT, atomic_json, file_sha256, within_project
from eventlog import event, stage

ENGINE = "b11576"
REPO = "unsloth/Qwen3.5-9B-GGUF"
REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
# The archive holds llama-<ENGINE>/llama-server next to its Metal dylibs (relative symlinks).
BINARY = f"bin/llama-{ENGINE}/llama-server"
FILES = {
    f"llama-{ENGINE}-bin-macos-arm64.tar.gz": "e46155b4dff22645df24cd46888e96fc46a417467b0c22983e58eae034fda81f",
    "Qwen3.5-9B-Q4_K_M.gguf": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
    "mmproj-F16.gguf": "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f",
}
FILE_BYTES = {
    f"llama-{ENGINE}-bin-macos-arm64.tar.gz": 12111998,
    "Qwen3.5-9B-Q4_K_M.gguf": 5680522464,
    "mmproj-F16.gguf": 918166080,
}


def extract_engine(archive: Path, binary_dir: Path):
    """The "data" filter refuses absolute paths, `..` and links leaving the folder; it keeps the owner's exec bit."""
    binary_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(binary_dir, filter="data")
    except tarfile.FilterError as error:
        raise RuntimeError("Chemin invalide dans l’archive native : " + str(error)) from error


def fetch(name, digest, target, cached, is_archive, hf_hub_download):
    """Write to a temporary sibling and publish only a verified file: an interruption never leaves `target`."""
    if cached.is_file() or is_archive:
        partial = target.with_name(target.name + ".tmp")
        try:
            if cached.is_file():
                shutil.copy2(cached, partial)
            else:
                urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/{ENGINE}/{name}", partial)
            if file_sha256(partial) != digest:
                raise RuntimeError(f"Empreinte invalide pour {name} ; le fichier ne sera pas utilisé.")
            os.replace(partial, target)
            return True
        finally:
            partial.unlink(missing_ok=True)
    # huggingface_hub downloads to its own .incomplete file and renames it when complete.
    hf_hub_download(REPO, name, revision=REVISION, local_dir=target.parent)
    return False


def download_native(config):
    if config["visionModel"] != "Qwen/Qwen3.5-9B":
        raise RuntimeError("Le moteur Metal natif validé utilise Qwen3.5-9B.")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download
    directory = within_project(config["nativeVisionDir"])
    directory.mkdir(parents=True, exist_ok=True)
    cached_bench = PROJECT_ROOT / ".bench-vision/llama"
    records = []
    for name, digest in FILES.items():
        is_archive = name.endswith(".tar.gz")
        folder = directory / ("downloads" if is_archive else "models")
        folder.mkdir(exist_ok=True)
        target = within_project(folder / name)
        cached = cached_bench / (name if is_archive else "models/" + name)
        with stage("models.native.file", "Préparation et vérification d’un fichier du moteur natif.", filename=name):
            verified = False
            if not target.exists():
                needed = cached.stat().st_size if cached.exists() else FILE_BYTES[name]
                if shutil.disk_usage(directory).free < needed + 1024 ** 3:
                    raise RuntimeError("Espace disque insuffisant pour le moteur vidéo natif.")
                verified = fetch(name, digest, target, cached, is_archive, hf_hub_download)
            if not verified and file_sha256(target) != digest:
                # Never keep a corrupted file: the next run downloads it again.
                target.unlink(missing_ok=True)
                raise RuntimeError(f"Empreinte invalide pour {name} ; le fichier ne sera pas utilisé.")
            if is_archive:
                extract_engine(target, within_project(directory / "bin"))
            records.append({"file": name, "sha256": digest, "bytes": target.stat().st_size})
    lock = {"engine": ENGINE, "modelRepo": REPO, "modelRevision": REVISION, "files": records}
    atomic_json(directory / "lock.json", lock)
    event("models.native.ready", "Moteur Metal natif et poids Qwen vérifiés.", engine=ENGINE, quantization="q4_k_m")
    return lock
