"""Install an isolated Apple Silicon (MPS) runtime inside the project folder; never change system Python."""
from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-system-torch", action="store_true", help="Reuse an existing PyTorch with MPS, without changing it.")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit("Python 3.11 à 3.13 requis : uv python install 3.12, puis relancer avec ce Python.")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Ce fork cible macOS sur Apple Silicon (arm64).")
    target = (ROOT / ".venv").resolve()
    if target.parent != ROOT or target.is_symlink():
        raise SystemExit("Le venv doit rester dans le dossier du projet.")
    # Pip also keeps its download cache inside the project folder.
    env = {**os.environ, "PIP_CACHE_DIR": str(ROOT / "cache/pip"), "PYTHONUTF8": "1"}
    venv.EnvBuilder(with_pip=True, system_site_packages=args.reuse_system_torch).create(target)
    python = target / "bin/python"
    def run(*arguments):
        subprocess.run([str(python), *arguments], cwd=ROOT, env=env, check=True)
    run("-m", "pip", "install", "--upgrade", "pip")
    if args.reuse_system_torch:
        run("-c", "import torch, torchaudio, torchvision; assert torch.backends.mps.is_available(), 'PyTorch avec MPS requis'")
    # PyPI's macOS arm64 wheels include MPS: torch comes from requirements.txt like everything else.
    run("-m", "pip", "install", "-r", str(ROOT / "worker/requirements.txt"))
    run("-c", "import torch; assert torch.backends.mps.is_available(), 'MPS indisponible : macOS 14+ requis'")
    print("Runtime installé dans .venv. Étape suivante : npm run models:download")


if __name__ == "__main__":
    main()
