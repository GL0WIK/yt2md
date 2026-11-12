# Banc d’essai vidéo isolé

Les scripts de ce dossier ne modifient pas la configuration, les dépendances ou les checkpoints de production. Ils utilisent Qwen3.5-9B, les mêmes 7 captures horodatées, le même contexte Parakeet, le même prompt et le même schéma JSON interne que le pipeline principal. Les essais Transformers utilisent les poids HF en NF4 ; les essais natifs utilisent une conversion GGUF Q4_K_M avec projecteur F16.

Les dépendances expérimentales sont installées dans `.bench-vision/deps`, exclusivement ajoutées au chemin Python du processus de test. Les captures sont copiées dans `.bench-vision/fixtures` avant le nettoyage des médias de production. Les caches de compilation et résultats restent dans `.bench-vision`.

```powershell
.venv\Scripts\python.exe -m pip install --target .bench-vision/deps --no-deps -r benchmarks/vision/requirements.txt
.venv\Scripts\python.exe -X utf8 benchmarks/vision/perf.py --prepare VIDEO_ID
.venv\Scripts\python.exe -X utf8 benchmarks/vision/perf.py --kernels-check
.venv\Scripts\python.exe -X utf8 benchmarks/vision/perf.py --profile reference --group 2 --runs 2
.venv\Scripts\python.exe -X utf8 benchmarks/vision/perf.py --profile fla --group 2 --runs 2
.venv\Scripts\python.exe -X utf8 benchmarks/vision/perf.py --profile fla-conv --group 2 --runs 2
```

`--free --max-tokens 256` désactive le décodeur JSON pour mesurer son coût. Ce mode est seulement un diagnostic et ses sorties peuvent ne pas respecter le contrat. `fla-compile` teste les accélérateurs avec le cache statique et `torch.compile`.

Mesurer sans autre inférence sur le GPU. Le premier passage peut inclure la compilation Triton ; comparer aussi les passages suivants. Les résultats enregistrent séparément le délai du premier token, le débit de décodage, le temps dans le décodeur JSON, le pic PyTorch et la validité du JSON. Les tests numériques couvrent l’attention récurrente, la préparation des séquences et la convolution, avec des tolérances adaptées au BF16.

## Moteur vidéo natif

```powershell
.venv\Scripts\python.exe -X utf8 benchmarks/vision/setup_llama.py
.venv\Scripts\python.exe -X utf8 benchmarks/vision/llama_perf.py --group 0 --runs 3
.venv\Scripts\python.exe -X utf8 benchmarks/vision/llama_perf.py --group 2 --runs 3
```

L’installation de test reste dans `.bench-vision/llama`, avec versions et empreintes fixées. Le serveur local de test est arrêté à la fin, même en cas d’erreur. Les JPEG sont assemblés dans un MP4 RGB sans perte à 1 fps, puis envoyés comme vidéo native avec JSON contraint et raisonnement désactivé. Les réponses complètes et mesures sont dans `.bench-vision/results`.

Après installation du moteur de production (`npm run models:download -- vision`), `adapter_check.py` teste le vrai adaptateur de la V2 sur les fixtures copiées, dont une dernière séquence d’une seule capture, puis vérifie le déchargement avant Parakeet. Son rapport reste dans le dossier de test.

Les résultats du 13 octobre sont documentés dans [../../docs/VISION_PERFORMANCE.md](../../docs/VISION_PERFORMANCE.md). Les essais de compilation forcée restent expérimentaux ; seul le moteur natif validé a été intégré à la V2.
