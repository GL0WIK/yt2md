# Tâches : fork YT2MD pour Mac M1 Pro 16 Go

Contexte et décisions : `tasks/plan.md`.

---

## Task 1 : fork public et branche de travail

**Description :** `gh repo fork lfpoulain/yt2md --remote` crée un fork public sous le compte de l'utilisateur et reconfigure les remotes (`origin` → fork, `upstream` → lfpoulain/yt2md). Ouvrir ensuite la branche `macos-apple-silicon`, qui sera fusionnée dans le `main` du fork au Checkpoint C.

**Critères d'acceptation :**
- [x] `git remote -v` montre `origin` (fork) et `upstream` (lfpoulain)
- [x] La branche `macos-apple-silicon` existe localement et sur le fork

**Vérification :**
- [x] `gh repo view <compte>/yt2md` répond
- [x] `git status` est propre sur la nouvelle branche

**Dépendances :** aucune (fork public validé par l'utilisateur).

**Fichiers :** aucun (git/gh uniquement).

**Taille :** XS

---

## Task 2 : runtime Python installable sur Mac, tests de référence

**Description :** `npm run python:setup` doit fonctionner sur macOS arm64. On installe torch/torchaudio/torchvision 2.11 depuis PyPI (roues MPS) au lieu de l'index `cu128`. La vérification `--reuse-system-torch` contrôle MPS au lieu de CUDA. `bitsandbytes` et `triton-windows` sont retirés des dépendances. On fixe ensuite l'état des tests existants sur Mac, comme base de comparaison.

**Critères d'acceptation :**
- [ ] `uv python install 3.12`, puis `uv run --python 3.12 --no-project python scripts/setup_runtime.py` crée `.venv` sans erreur
- [ ] Dans le venv : `python -c "import torch; assert torch.backends.mps.is_available()"` passe
- [ ] Plus aucune mention de `cu128` ou de CUDA dans `setup_runtime.py` et `requirements.txt`

**Vérification :**
- [ ] `source .venv/bin/activate && npm install && npm test` : résultat noté (référence)
- [ ] `npm run test:python` : résultat noté, avec la liste des échecs propres à macOS, s'il y en a
- [ ] `npm run typecheck` passe

**Dépendances :** T1

**Fichiers :**
- `scripts/setup_runtime.py`
- `worker/requirements.txt` (suppression de `bitsandbytes` et `triton-windows`)

**Taille :** S

---

## Task 3 : CUDA → MPS dans config, worker et doctor

**Description :** `config.ts` ne lit plus `DEVICE` et fixe `device: "mps"`. `ASR_COMPILE` vaut `false` par défaut (pas de Triton sur macOS). Dans le worker, les `torch.cuda.*` deviennent leurs équivalents `torch.mps.*` (`empty_cache`, `current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory`) dans `gpu_fields`, `unload`, `load` et `doctor`. `doctor` vérifie `torch.backends.mps.is_available()`. Le backend vision `transformers` en NF4 (bitsandbytes) n'existe plus : seul `llama_cpp` reste, avec BF16 en option.

**Critères d'acceptation :**
- [ ] `config.ts` : plus de `DEVICE` ni de validation `cuda:N`. `VISION_QUANTIZATION` n'accepte plus `nf4`
- [ ] `npm run doctor` sur le Mac renvoie `gpu` renseigné (Apple M1 Pro, mémoire) et ne signale plus d'erreur CUDA, Triton ou bitsandbytes. Il ne doit rester que « modèles absents »
- [ ] `grep -rn "torch.cuda" worker scripts` ne renvoie rien

**Vérification :**
- [ ] Tests existants adaptés : les faux `torch.cuda` deviennent de faux `torch.mps` (`test_logging.py`, `test_vision_generation.py`, `test_vision_monitor.py`)
- [ ] `npm test && npm run test:python && npm run typecheck`

**Dépendances :** T2

**Fichiers :**
- `server/src/config.ts`
- `worker/runtime.py`
- `worker/tests/test_logging.py`, `worker/tests/test_vision_generation.py`
- `scripts/smoke_asr.py`, `scripts/smoke_vision.py` (device en dur)

**Taille :** M

---

## Task 4 : Qwen3-ASR et aligneur réels sur MPS

**Description :** télécharger uniquement les modèles audio, puis transcrire un vrai extrait sur MPS. On choisit le dtype (BF16 ou FP16 de repli sur MPS) et on active `PYTORCH_ENABLE_MPS_FALLBACK=1` si un opérateur manque. On mesure le facteur temps réel et la mémoire de pointe. C'est la tâche la plus risquée : si elle échoue, on revoit le plan avant d'aller plus loin.

**Critères d'acceptation :**
- [ ] `npm run models:download -- asr aligner` réussit
- [ ] `scripts/smoke_asr.py` sur un extrait français de 1 à 2 min produit un texte cohérent avec des horodatages par mot
- [ ] Mesures notées dans le plan : durée et facteur temps réel, mémoire de pointe, dtype retenu

**Vérification :**
- [ ] `npm run test:python`
- [ ] Contrôle manuel : relecture de la transcription et des horodatages (aucun mot à 0 s, ordre croissant)

**Dépendances :** T3

**Fichiers :**
- `worker/runtime.py` (dtype et éventuel repli selon le device)
- `worker/main.py` (variable `PYTORCH_ENABLE_MPS_FALLBACK`, si elle est nécessaire)
- `scripts/smoke_asr.py` (s'il force `cuda`)

**Taille :** S–M

---

## Checkpoint A : transcription sur Mac
- [ ] `npm test`, `npm run test:python` et `npm run typecheck` passent
- [ ] `npm run doctor` : seul le modèle vision manque
- [ ] Facteur temps réel de l'ASR jugé acceptable par l'utilisateur
- [ ] **Revue avec l'utilisateur avant la phase 2**

---

## Task 5 : téléchargement vérifié de llama.cpp macOS arm64

**Description :** vérifier d'abord sur la page de release `b11576` le nom exact, le format et le SHA-256 de l'archive `macos-arm64`. Dans `native_download.py`, cette archive remplace les deux zips `win-cuda` (plus de `cudart`) ; le GGUF et le projecteur restent les mêmes. On supprime le contrôle `os.name != "nt"`, puis on extrait l'archive, on rend `llama-server` exécutable et on vérifie que Gatekeeper ne le bloque pas.

**Critères d'acceptation :**
- [ ] `npm run models:download -- vision` sur le Mac télécharge, vérifie les SHA-256 et écrit `models/native-vision/lock.json`
- [ ] `models/native-vision/bin/.../llama-server --version` s'exécute

**Vérification :**
- [ ] Test unitaire : extraction de l'archive, bit exécutable, refus d'un chemin hors du dossier
- [ ] `npm run test:python`

**Dépendances :** T2

**Fichiers :**
- `worker/native_download.py`
- `worker/tests/test_native_vision.py`

**Taille :** S

---

## Task 6 : moteur vision Metal et garde mémoire unifiée

**Description :** `native_lock` pointe sur `bin/llama-server`. `start()` passe `--device MTL0` ou omet `--device`. On vérifie que `-fa on` et les options `--video-*` sont acceptés par la build Metal. `LocalRuntime.load("vision")` n'appelle plus `torch.cuda.set_device`, et le Job Object Windows (`bind_to_worker`, `release_job`) est supprimé. `check_vision_memory` utilise la mémoire système disponible (`vm_stat`) au lieu de `nvidia-smi`, avec un budget d'environ 8 Gio pour le Q4_K_M.

**Critères d'acceptation :**
- [ ] `scripts/smoke_vision.py` sur un groupe de 7 captures renvoie un JSON valide, sur Metal
- [ ] Si la mémoire disponible est trop basse, le chargement est refusé avec un message clair (en français) au lieu de faire swapper la machine
- [ ] Après `unload`, le processus `llama-server` est arrêté (pas de processus orphelin, vérifié avec `pgrep llama-server`)

**Vérification :**
- [ ] Tests unitaires : arguments du moteur, garde mémoire avec un faux `vm_stat` (les tests `nvidia-smi` sont supprimés)
- [ ] `npm run test:python`
- [ ] Débit noté (tokens/s et secondes par groupe)

**Dépendances :** T5 (et T3 pour le device `mps`)

**Fichiers :**
- `worker/native_vision.py`
- `worker/vision_monitor.py`
- `worker/runtime.py` (branche vision de `load`)
- `worker/tests/test_vision_monitor.py`, `worker/tests/test_native_vision.py`

**Taille :** M

---

## Task 7 : traitement complet d'une vidéo depuis l'interface

**Description :** sur le Mac, avec `npm run dev`, ajouter une vidéo courte (3 à 5 min), la transcrire et vérifier les trois exports. Dans `media.py`, on supprime le chemin `-hwaccel cuda`/`scale_cuda` et son repli : le décodage passe directement par le CPU. On teste aussi pause et reprise.

**Critères d'acceptation :**
- [ ] `outputs/video--ID.md`, `.json` et `.local.json` sont produits. Le `.json` est valide selon `docs/corpus.schema.json`
- [ ] Pause pendant la vision puis reprise : aucun travail audio n'est refait
- [ ] Mesures notées dans le plan : durée totale par minute de vidéo, mémoire de pointe, pression mémoire (Moniteur d'activité)

**Vérification :**
- [ ] `ffmpeg -encoders | grep libx264rgb` (nécessaire au clip natif)
- [ ] `npm run smoke:media`
- [ ] Contrôle manuel dans l'interface : progression, journal et téléchargement ZIP

**Dépendances :** T4, T6

**Fichiers :**
- `worker/media.py` (suppression du chemin GPU)
- `worker/tests/test_robustness.py` (suppression du test de repli NVDEC)

**Taille :** S

---

## Checkpoint B : pipeline complet sur Mac
- [ ] Tous les tests passent (`npm run check`)
- [ ] Une vidéo réelle est traitée de bout en bout sur le Mac
- [ ] `grep -rniE "cuda|nvidia|\.exe" worker scripts server/src` ne renvoie que des libellés à nettoyer en T8
- [ ] **Revue avec l'utilisateur : la vitesse mesurée est-elle acceptable ?** (sinon, T9 passe à un modèle plus petit)

---

## Task 8 : documentation, `.env.example` et messages

**Description :** réécrire le README pour macOS Apple Silicon : badges (macOS, Apple Silicon, Metal au lieu de Windows/CUDA), configuration requise (16 Go de mémoire unifiée), installation (`uv python install 3.12`, activation du venv, commandes en zsh au lieu de PowerShell), performances mesurées en T7. Dans `.env.example` : retirer `DEVICE`, `VISION_QUANTIZATION=nf4` et les chemins `.exe`, passer `ASR_COMPILE=false`, et adapter la note sur les cookies (Safari/Chrome sous macOS). Les messages « CUDA natif », « VRAM » et « GPU » deviennent « moteur Metal » et « mémoire ».

**Critères d'acceptation :**
- [ ] Une installation depuis zéro en suivant uniquement le README réussit sur le Mac
- [ ] Plus aucune mention de Windows, CUDA ou PowerShell dans le README ni dans `.env.example`

**Vérification :**
- [ ] `npm run format:check`
- [ ] Relecture du README par l'utilisateur

**Dépendances :** T7

**Fichiers :**
- `README.md`
- `.env.example`
- `worker/native_vision.py`, `worker/runtime.py` (libellés uniquement)

**Taille :** S

---

## Task 9 : (conditionnelle) modèle vision plus petit

**Description :** à faire seulement si les mesures de T7 sont jugées trop lentes au Checkpoint B. L'utilisateur préfère un modèle plus petit à des réglages qui dégradent l'image. On passe donc à un GGUF vision plus petit de la même famille (Qwen3.5-4B par exemple, à vérifier sur Hugging Face), avec un nouveau verrou SHA-256 pour le modèle et son projecteur. On adapte la vérification `visionModel` dans `config.ts` et `native_vision.py`, et le budget mémoire. Le cache vision est invalidé, ce qui est attendu.

**Critères d'acceptation :**
- [ ] Le modèle retenu est documenté avec avant/après (durée par minute, pression mémoire, qualité des descriptions sur une même vidéo)
- [ ] Le JSON reste valide sur 3 groupes de test

**Vérification :**
- [ ] `npm run check`
- [ ] Nouvelle passe de T7 sur la même vidéo

**Dépendances :** Checkpoint B

**Fichiers :** `worker/native_download.py`, `worker/native_vision.py`, `worker/vision_monitor.py`, `server/src/config.ts`, `.env.example`

**Taille :** S–M

---

## Checkpoint C : terminé
- [ ] Tous les critères d'acceptation sont cochés
- [ ] `npm run check` passe sur le Mac
- [ ] Branche fusionnée dans le `main` du fork public et poussée (pas de PR vers lfpoulain)
