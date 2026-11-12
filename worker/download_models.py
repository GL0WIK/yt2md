"""The only network-enabled model operation. Inference loads these snapshots offline."""
from __future__ import annotations

from datetime import datetime, timezone
import fnmatch
import os
import shutil

from common import PROJECT_ROOT, atomic_json, read_json, within_project
from eventlog import event, stage

PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]
# Duplicate FP32 shards, when a repository ships them, are never loaded by BF16 inference.
IGNORED = ["*fp32*"]


def wanted(filename: str, patterns: list[str]) -> bool:
    return (any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)
            and not any(fnmatch.fnmatch(filename, pattern) for pattern in IGNORED))


def download_models(request: dict) -> dict:
    config = request["config"]
    directory = within_project(config["modelsDir"])
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(directory / ".hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if config.get("visionBackend") == "llama_cpp":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import HfApi, snapshot_download
    lock_path = directory / "models.lock.json"
    lock = read_json(lock_path) or {"models": {}}
    model_ids = {"asr": config["asrModel"], "aligner": config["asrAlignerModel"], "vision": config["visionModel"]}
    roles = request.get("roles", list(model_ids))
    if not roles or any(role not in model_ids for role in roles):
        raise ValueError("Modèles possibles : asr, aligner, vision.")
    for role in roles:
        model_id = model_ids[role]
        native = role == "vision" and config.get("visionBackend") == "llama_cpp"
        selected_patterns = [p for p in PATTERNS if p != "*.safetensors"] if native else PATTERNS
        previous = lock["models"].get(role, {})
        revision = request.get("revisions", {}).get(role)
        if not revision and not request.get("refresh") and previous.get("modelId") == model_id:
            revision = previous.get("revision")
        with stage("models.resolve", "Vérification de la révision et de la taille des fichiers.", role=role, modelId=model_id):
            info = HfApi().model_info(model_id, revision=revision or "main", files_metadata=True)
        revision = info.sha
        folder = within_project(directory / role / revision)
        needed = sum(file.size or 0 for file in info.siblings if wanted(file.rfilename, selected_patterns))
        present = sum(file.stat().st_size for file in folder.rglob("*.safetensors") if wanted(file.name, selected_patterns)) if folder.exists() else 0
        if shutil.disk_usage(directory).free < max(0, needed - present) + 2 * 1024 ** 3:
            raise RuntimeError(f"Espace disque insuffisant pour {model_id} : environ {needed / 1024 ** 3:.1f} Gio de fichiers.")
        event("models.download.plan", "Téléchargement des poids du modèle.", role=role, modelId=model_id,
              revision=revision, modelBytes=needed, presentWeightBytes=present, diskFreeBytes=shutil.disk_usage(directory).free)
        with stage("models.download", "Téléchargement du modèle local.", role=role, modelId=model_id):
            snapshot_download(repo_id=model_id, revision=revision, local_dir=str(folder),
                              cache_dir=str(directory / ".hub"), allow_patterns=selected_patterns,
                              ignore_patterns=IGNORED, max_workers=4)
        if not (folder / "config.json").is_file() or (not native and not list(folder.glob("*.safetensors"))):
            raise RuntimeError(f"Téléchargement incomplet pour {model_id}.")
        lock["models"][role] = {"modelId": model_id, "revision": revision,
                                "path": folder.relative_to(PROJECT_ROOT).as_posix(),
                                "downloadedAt": datetime.now(timezone.utc).isoformat()}
        atomic_json(lock_path, lock)
        event("models.download.complete", "Poids téléchargés et révision enregistrée.", role=role, modelId=model_id, revision=revision)
        if native:
            from native_download import download_native
            download_native(config)
    return lock
