# Plan : fork YT2MD pour MacBook Pro M1 Pro 16 Go

## Objectif

Faire tourner le pipeline complet (téléchargement → transcription Qwen3-ASR → analyse visuelle Qwen3.5-9B → exports) sur un Mac Apple Silicon M1 Pro avec 16 Go de mémoire unifiée, sans GPU NVIDIA.

## État des lieux (lu dans le code)

| Endroit | Ce qui bloque sur Mac |
| --- | --- |
| `scripts/setup_runtime.py:31-33` | installe PyTorch depuis l'index `cu128` et exige `torch.cuda.is_available()` |
| `worker/requirements.txt` | `bitsandbytes` (CUDA), `triton-windows` (déjà réservé à `win32`) |
| `server/src/config.ts:72,76` | `DEVICE` par défaut `cuda:0`, et toute autre valeur est refusée |
| `worker/runtime.py` (`gpu_fields`, `doctor`, `unload`, `load`) | appels directs à `torch.cuda.*` |
| `worker/vision_monitor.py` | `torch.cuda.mem_get_info` + `nvidia-smi` |
| `worker/native_download.py:13-25,51` | archives llama.cpp `win-cuda` uniquement, et `os.name != "nt"` → erreur |
| `worker/native_vision.py:123,170` | `bin/llama-server.exe`, `--device CUDA0` |
| `worker/media.py:226,262` | décodage `-hwaccel cuda` (il existe déjà un repli CPU) |
| `package.json` | les scripts appellent `python`, absent du Mac (seul 3.15 est installé, hors plage 3.11–3.13) |

Ce qui marche déjà : Node 26.8, FFmpeg 9, yt-dlp, uv. `killProcessTree` et `config.ts` gèrent déjà POSIX (`bin/python`).

## Décisions d'architecture

1. **Fork public sous ton compte, macOS uniquement, sans PR upstream.** On *remplace* CUDA par MPS/Metal au lieu d'ajouter une branche : on supprime le code propre à Windows/NVIDIA (index `cu128`, `triton-windows`, `bitsandbytes`, archives `win-cuda` + `cudart`, `llama-server.exe`, Job Object `bind_to_worker`, `nvidia-smi`, `-hwaccel cuda`). Fusionner plus tard les nouveautés de lfpoulain reste possible, mais avec des conflits acceptés.
2. **Device fixe `mps`.** La variable `DEVICE` disparaît de `.env` et `config.ts` fixe la valeur interne à `mps`. Les appels `torch.cuda.*` deviennent `torch.mps.*`, et la mémoire libre se lit via `vm_stat`.
3. **Transcription : PyTorch MPS** (roues PyPI standard pour arm64), BF16 si MPS le supporte, sinon FP16. `ASR_COMPILE=false` sur Mac : Triton n'existe pas sur macOS et le code sait déjà s'en passer (`compile_available()`).
4. **Analyse visuelle : llama.cpp Metal**, même release `b11576`, archive `macos-arm64` épinglée par SHA-256 comme sur Windows. On garde le même GGUF Q4_K_M et le même projecteur (déjà multiplateformes), donc le moteur et les clés de cache restent identiques.
5. **Mémoire unifiée de 16 Go :** un seul gros modèle chargé à la fois, ce que le code fait déjà (`LocalRuntime.unload`). Pour le contrôle avant chargement, on remplace la VRAM CUDA par la mémoire système disponible (`vm_stat`), avec un budget d'environ 8 Gio pour Qwen3.5-9B Q4.
6. **Décodage vidéo sur CPU** (le chemin de repli existant devient le seul chemin). VideoToolbox est remis à plus tard, à faire seulement si les mesures montrent que c'est le goulot.
8. **Vitesse insuffisante → modèle vision plus petit** plutôt que garder Qwen3.5-9B (choix de l'utilisateur). On passe par T9, selon les mesures de T7.
7. **Python 3.12 via uv**, sans modifier le Python système. On documente `uv python install 3.12`, puis le venv activé fournit `python` à tous les scripts npm.

## Graphe de dépendances

```
T1 Fork + branche
  └── T2 Runtime Python sur Mac (venv 3.12, torch MPS, deps)  ← base de référence des tests
        ├── T3 Device mps (config.ts + utilitaire worker + doctor)
        │     └── T4 Transcription réelle sur MPS          ── Checkpoint A
        │           └── T7 Pipeline complet de bout en bout
        └── T5 Téléchargement llama.cpp Metal
              └── T6 Moteur vision Metal + garde mémoire
                    └── T7 (ci-dessus)                      ── Checkpoint B
                          ├── T8 Docs + .env.example + messages
                          └── T9 (conditionnel) Réglages perf 16 Go ── Checkpoint C
```

T3/T4 et T5/T6 sont indépendants et peuvent avancer en parallèle après T2.

## Liste des tâches

Le détail de chaque tâche (critères, vérification, fichiers) est dans `tasks/todo.md`.

### Phase 0 : socle
- [x] T1 : fork public et branche `macos-apple-silicon`
- [x] T2 : runtime Python installable sur Mac et tests de référence

### Phase 1 : transcription (risque le plus élevé, traitée en premier)
- [ ] T3 : CUDA → MPS dans config, worker et doctor
- [ ] T4 : Qwen3-ASR + aligneur réels sur MPS

### Checkpoint A : la transcription tourne sur le Mac

### Phase 2 : analyse visuelle
- [ ] T5 : téléchargement vérifié de llama.cpp macOS arm64
- [ ] T6 : moteur vision Metal et garde mémoire unifiée
- [ ] T7 : traitement complet d'une vidéo depuis l'interface

### Checkpoint B : pipeline complet sur Mac

### Phase 3 : finition
- [ ] T8 : README, `.env.example`, messages et doctor adaptés
- [ ] T9 : (conditionnel) modèle vision plus petit si T7 est trop lent

### Checkpoint C : prêt à l'usage

## Risques et parades

| Risque | Impact | Parade |
| --- | --- | --- |
| `transformers` Qwen3-ASR utilise un opérateur absent de MPS | Élevé | T4 tôt. `PYTORCH_ENABLE_MPS_FALLBACK=1` pour basculer les opérateurs manquants sur CPU. En dernier recours, ASR sur CPU (lent) |
| BF16 lent ou instable sur M1 via MPS | Moyen | Repli FP16 dans T4. FP16 ne change pas le cache ASR, qui hache les modèles et la config, pas le dtype : à vérifier, sinon invalider proprement |
| L'archive llama.cpp `b11576` macOS n'existe pas, ou change de nom ou de format (`.zip` ou `.tar.gz`) | Moyen | T5 vérifie la page de release avant de coder. Sinon, prendre la release la plus proche publiant `macos-arm64` et l'épingler (cela change le moteur, donc invalide le cache vision Mac uniquement) |
| `zipfile` perd le bit exécutable, et Gatekeeper peut bloquer le binaire | Moyen | `chmod +x` après extraction. Un fichier téléchargé par urllib n'a pas l'attribut de quarantaine, à vérifier dans T5 |
| 16 Go : macOS + navigateur + Qwen 9B Q4 (~7 Go avec contexte) → swap | Moyen | Garde mémoire dans T6. T9 : réduire `-c`, `VISION_MAX_PIXELS` ou passer à un modèle plus petit |
| Vitesse : environ 5 à 10× plus lent qu'une RTX 4090, la vision peut être plus lente que le temps réel | Moyen | Mesures dans T7. Si c'est trop lent, T9 passe à un GGUF vision plus petit |
| `-fa on` (Flash Attention) ou `--video-*` non supportés par la build Metal | Faible | Vérifié dans T6. Sinon, ajuster `SERVER_ARGS` par plateforme, ce qui change la clé de cache vision |
| Fusions futures depuis lfpoulain conflictuelles (code CUDA supprimé) | Faible | Accepté : fork Mac indépendant, pas de PR upstream |

## Décisions de l'utilisateur (2026-11-19)

1. Fork **public** de lfpoulain/yt2md sous le compte de l'utilisateur.
2. **macOS uniquement** : la compatibilité Windows/CUDA est abandonnée.
3. Si la vision est trop lente : **modèle plus petit**.
4. **Pas de PR** vers le projet d'origine.
