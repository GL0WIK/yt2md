"""Generation progress without logging model output or synchronizing CUDA for metrics."""
from __future__ import annotations

import time
import subprocess

from eventlog import event


def physical_free_memory(torch, device):
    # Under Windows WDDM, CUDA's budget can differ from physical free VRAM.
    # Select by UUID, not ordinal: CUDA_VISIBLE_DEVICES can remap GPU indices.
    try:
        identifier = str(torch.cuda.get_device_properties(device).uuid)
        if not identifier.startswith(("GPU-", "MIG-")):
            identifier = "GPU-" + identifier
        result = subprocess.run(
            ["nvidia-smi", f"--id={identifier}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(result.stdout.strip()) * 1024 ** 2
    except (AttributeError, ValueError, OSError, subprocess.SubprocessError):
        return None


def check_vision_memory(torch, config):
    # Conservative budgets including CUDA/workspace headroom, not guarantees.
    budgets = {"Qwen/Qwen3.8-27B": 20, "Qwen/Qwen3.5-9B": 10}
    budget = budgets.get(config["visionModel"])
    if budget is None or config["visionQuantization"] != "nf4":
        return
    free, total = torch.cuda.mem_get_info(config["device"])
    physical_free = physical_free_memory(torch, config["device"])
    available = min(free, physical_free) if physical_free is not None else free
    required = budget * 1024 ** 3
    event("vision.memory.check", "Vérification de la mémoire disponible avant chargement de Qwen.",
          vramFreeBytes=available, cudaFreeBytes=free, physicalFreeBytes=physical_free,
          vramTotalBytes=total, estimatedRequiredBytes=required)
    if available < required:
        raise RuntimeError(
            f"{config['visionModel']} : seulement {available / 1024 ** 3:.1f} Gio de VRAM disponibles ; "
            f"le budget prudent est de {budget} Gio pour ce réglage. Fermer les applications utilisant "
            "le GPU, puis reprendre le traitement. Les morceaux audio terminés restent en cache."
        )


class GenerationProgress:
    """Transformers streamer protocol; the initial put contains the whole prompt."""

    def __init__(self):
        self.started = time.perf_counter()
        self.prompt_seen = False
        self.tokens = 0
        self.first_token = None
        self.last_token = None

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        now = time.perf_counter()
        count = int(value.shape[-1]) if len(value.shape) > 1 else 1
        self.tokens += count
        self.last_token = now
        if self.first_token is None:
            self.first_token = now
            event("vision.generate.first_token", "Premier token produit par Qwen.", **self.fields())

    def fields(self):
        now = time.perf_counter()
        fields = {"generatedTokens": self.tokens, "waitingForFirstToken": self.first_token is None}
        if self.first_token is not None:
            fields["firstTokenMs"] = round((self.first_token - self.started) * 1000)
            fields["tokensPerSecond"] = round((self.tokens - 1) / max(now - self.first_token, 0.001), 2)
            fields["lastTokenAgeMs"] = round((now - self.last_token) * 1000)
        return fields

    def end(self):
        event("vision.generate.stream_complete", "Génération Qwen terminée ; validation du JSON.", **self.fields())
