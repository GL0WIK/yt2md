# Mesures vidéo indépendantes — 13 octobre 2026

> Mesures réalisées lorsque la transcription utilisait encore Parakeet ; le moteur vision et ses réglages sont inchangés depuis le passage à Qwen3-ASR.

Le moteur choisi atteint **120–122 tokens/s** sur les extraits testés, avec un JSON valide, une vidéo de sept captures et le contexte Parakeet. Il est activé par défaut après les essais isolés et la validation de son adaptateur. Le modèle reste Qwen3.5-9B ; ses poids passent de la quantification HF NF4 à une conversion GGUF Q4_K_M.

## Protocole

- Machine détectée : **RTX 4090, 24 Go**, Windows, Python 3.12.10, PyTorch 2.11.0+cu128, Transformers 5.19.0.
- Vidéo `tnf26IU_7tg` : groupes 0 (0–7 s) et 2 (14–21 s), sept JPEG à 1 fps, contexte audio Parakeet conservé et horodatages source.
- Même consigne `local-observed-video-v2-compact`, mêmes quatre champs JSON et limites internes ; température 0,7, top-p 0,8, top-k 20, pénalité de répétition 1,1, plafond 1 024 tokens.
- Pas de raisonnement avec le moteur retenu. Pas de réutilisation de prompt dans les essais natifs (`cache_prompt=false`, `cache_n=0`). Trois graines par groupe : 42, 43 et 44.
- Les dépendances expérimentales et caches de compilation restent dans `.bench-vision`. Les captures ont été copiées avant les essais ; aucun checkpoint de production n’est écrit par les benchmarks.

Le débit désigne le décodage des tokens de sortie. Le temps par groupe inclut le traitement visuel et l’appel d’inférence, hors chargement du modèle. La première exécution du programme peut en plus initialiser le runtime et compiler les kernels.

## Comparaison

| Moteur / essai                                                          |              Débit observé | JSON            | Décision             |
| ----------------------------------------------------------------------- | -------------------------: | --------------- | -------------------- |
| Transformers NF4 de référence, groupe 2, passage valide                 |             13,59 tokens/s | Valide          | Référence            |
| Transformers + FLA, passage valide                                      |             15,06 tokens/s | Valide          | Gain limité          |
| Transformers + kernels FLA, convolution et normalisation, passage chaud |             16,04 tokens/s | Valide          | Gain limité          |
| Compilation forcée expérimentale + Xgrammar, passage chaud              |             36,24 tokens/s | Valide          | Non retenu           |
| Compilation forcée, génération sans contrainte JSON, passage chaud      |             82,78 tokens/s | Invalide        | Diagnostic seulement |
| llama.cpp CUDA + GGUF Q4_K_M + JSON contraint, six passages             | **119,90–121,51 tokens/s** | **6/6 valides** | **Retenu**           |

La compilation automatique était désactivée par le quantificateur bitsandbytes. Forcer ce chemin dans le processus de test a nécessité une modification expérimentale de sa déclaration de compatibilité ; elle n’a pas été intégrée en production. Le passage compilé à froid avec Xgrammar a pris environ 66 secondes. Les masques Xgrammar transférés de façon asynchrone n’ont pas amélioré les passages chauds (33–35 tokens/s).

Les deux premiers profils Transformers ont aussi produit une réponse JSON invalide avec la graine 42 ; leur chiffre ci-dessus correspond au passage valide suivant. Les tests numériques des kernels FLA et de normalisation ont passé les tolérances BF16, mais leur gain ne justifiait pas leur déploiement. Le résultat libre à 82,78 tokens/s ne remplit pas le besoin d’export structuré.

## Moteur retenu : six essais

| Groupe | Graine | Tokens de sortie |           Débit | Temps par groupe | JSON   |
| ------ | -----: | ---------------: | --------------: | ---------------: | ------ |
| 0      |     42 |              170 | 120,70 tokens/s |          2,569 s | Valide |
| 0      |     43 |              160 | 120,22 tokens/s |          2,397 s | Valide |
| 0      |     44 |              136 | 119,90 tokens/s |          2,163 s | Valide |
| 2      |     42 |              188 | 121,35 tokens/s |          2,550 s | Valide |
| 2      |     43 |              137 | 120,52 tokens/s |          2,055 s | Valide |
| 2      |     44 |              204 | 121,51 tokens/s |          2,629 s | Valide |

À environ 120 tokens/s contre 13,59, le débit de décodage augmente d’environ **8,9 fois** sur cette comparaison. Les six réponses et mesures brutes sont conservées dans `.bench-vision/results/llama-video-g*-r*.json`. Le serveur reçoit un MP4 RGB sans perte assemblé à partir des JPEG, via une entrée `input_video`, et une grammaire JSON générée à partir du schéma interne.

## Validation de l’adaptateur V2

Le vrai `LocalRuntime`, avec les fichiers installés sous `models/native-vision`, a ensuite traité :

- Sept captures du groupe 0 : JSON valide, 121,10 tokens/s ; 7,611 s en incluant le démarrage du moteur.
- Sept captures du groupe 2 : JSON valide, 121,52 tokens/s ; 2,459 s.
- Un groupe final d’une seule capture : JSON valide, 121,50 tokens/s ; 1,126 s.
- Passage à Parakeet : arrêt confirmé du processus natif avant le chargement audio ; allocation PyTorch nulle avant ce chargement. Parakeet a chargé puis déchargé correctement.

Résultats : `.bench-vision/results/adapter-check.json`. Le serveur est configuré avec `VISION_BACKEND=llama_cpp`. Le moteur est lancé uniquement pour l’étape visuelle, sur une adresse locale avec une clé temporaire, puis arrêté lors d’un changement de modèle ou d’une pause. Les anciens checkpoints audio restent compatibles ; les checkpoints visuels sont invalidés par le changement de moteur. Le JSON d’export conserve le contrat 2.0 du projet original.

## Versions et limites

Moteur officiel [llama.cpp b11576](https://github.com/ggml-org/llama.cpp/releases/tag/b11576), CUDA Windows 12.4, GPU intégral, Flash Attention activée, contexte 8 192, une seule séquence, raisonnement désactivé. Conversion [Unsloth Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5), révision `3885219b6810b007914f3a7950a8d1b469d598a5`, poids `Qwen3.5-9B-Q4_K_M.gguf` et projecteur `mmproj-F16.gguf`. L’installateur vérifie les SHA-256 ; le verrou est dans `models/native-vision/lock.json`.

Ce changement conserve les pixels des captures, le contexte audio et le contrat de sortie, mais **n’est pas une comparaison numérique à quantification identique** : NF4 devient Q4_K_M, le projecteur est en F16 et les processeurs vidéo diffèrent. Le moteur natif compte 3 231 et 3 170 tokens d’entrée pour les deux groupes, contre 1 920 avec Transformers pour le groupe 2. Le texte produit peut donc varier.

Ces essais valident le débit et le format, pas l’exactitude visuelle. Certaines réponses comportent encore des erreurs de marque ou de description ; une évaluation annotée serait nécessaire pour comparer la qualité. Deux courts extraits ne permettent pas de garantir le débit moyen de 400 vidéos, ni d’assurer que chaque réponse tient dans la limite de tokens. Les JSON incomplets restent refusés et bénéficient de la seconde tentative prévue par le pipeline.
