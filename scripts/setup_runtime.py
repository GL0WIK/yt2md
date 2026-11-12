"""Install an isolated CUDA runtime inside the project folder; never change system Python."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-system-torch", action="store_true", help="Reuse an existing compatible CUDA PyTorch, without changing it.")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit("Python 3.11 à 3.13 requis ; Python 3.12 recommandé sous Windows.")
    target = (ROOT / ".venv").resolve()
    if target.parent != ROOT or target.is_symlink():
        raise SystemExit("Le venv doit rester dans le dossier du projet.")
    # Pip also keeps its download cache inside the project folder.
    env = {**os.environ, "PIP_CACHE_DIR": str(ROOT / "cache/pip"), "PYTHONUTF8": "1"}
    venv.EnvBuilder(with_pip=True, system_site_packages=args.reuse_system_torch).create(target)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    def run(*arguments):
        subprocess.run([str(python), *arguments], cwd=ROOT, env=env, check=True)
    run("-m", "pip", "install", "--upgrade", "pip")
    if args.reuse_system_torch:
        run("-c", "import torch, torchaudio, torchvision; assert torch.cuda.is_available() and torch.version.cuda and tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12,8), 'PyTorch CUDA >= 12.8 requis'")
    else:
        run("-m", "pip", "install", "torch==2.11.0", "torchaudio==2.11.0", "torchvision==0.26.0", "--index-url", "https://download.pytorch.org/whl/cu128")
    run("-m", "pip", "install", "-r", str(ROOT / "worker/requirements.txt"))
    print("Runtime installé dans .venv. Étape suivante : npm run models:download")


if __name__ == "__main__":
    main()
