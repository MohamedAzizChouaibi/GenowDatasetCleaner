# GenowDatasetCleaner

**GenowDatasetCleaner** est une application de bureau (Python + Tkinter) conçue pour nettoyer et organiser des datasets d'images au format **YOLOv11 / YOLOv12** (Ultralytics). Elle détecte les images de mauvaise qualité (floues, sur/sous-exposées, peu informatives) et les doublons à l'aide d'embeddings visuels (**CLIP** par défaut, plusieurs autres VLM disponibles), puis permet de les supprimer en conservant les fichiers de labels YOLO (`.txt`) associés.

Un dossier est reconnu comme dataset YOLO s'il contient soit un `data.yaml` avec les clés `train:` et `names:`/`nc:`, soit des répertoires parallèles `images/` et `labels/`.

## Fonctionnalités

*   **Détection de doublons**: pré-passe par hachage perceptuel (pHash) pour les doublons exacts, puis clustering par similarité cosinus sur les embeddings VLM.
*   **Embedders multiples** (sélectionnables dans l'interface ou via `--embedder`):
    *   **CLIP** ViT-B/32, ViT-B/16, ViT-L/14 (OpenAI, par défaut).
    *   **OpenCLIP** (nécessite `open_clip_torch`).
    *   **SigLIP** et **DINOv2** (nécessitent `transformers`).
    *   **API hébergées** (nécessitent une clé API + `requests`): Jina CLIP v2, Voyage multimodal-3, et VLM par légende — Qwen-VL, OpenAI GPT-4o-mini, Claude, Gemini.
*   **Analyse de qualité**:
    *   Images **floues** (variance du Laplacien).
    *   Images **trop sombres** / **trop claires** (luminosité moyenne).
    *   Images à **faible information** (écart-type des pixels).
*   **Filtre par nom**: signale les fichiers contenant les tokens `det` / `seg`.
*   **Suppression sûre**: corbeille système (`send2trash`) ou dossier de quarantaine, avec **annulation** (undo) de la dernière suppression.
*   **Interface graphique** (Tkinter) et **mode CLI** sans affichage.
*   **Rapport JSON** généré automatiquement après chaque analyse.

## Dépendances

Listées dans `requirements.txt`.

**Obligatoires**:

*   `numpy`
*   `opencv-python` (cv2)
*   `Pillow` (PIL)
*   `torch`
*   `send2trash` — suppression vers la corbeille système
*   `imagehash` — pré-passe par hachage perceptuel
*   `clip` (installé depuis `git+https://github.com/openai/CLIP.git`)

`tkinter` est requis pour l'interface graphique (inclus avec Python sur la plupart des plateformes ; sous Debian/Ubuntu : `sudo apt install python3-tk`).

**Optionnelles** (installer uniquement les backends souhaités) :

*   `open_clip_torch` — embedders `openclip:*`
*   `transformers` — embedders `siglip:*` / `dinov2:*` et embedders par légende (API)
*   `requests` — tout embedder via API (Jina, Voyage, Qwen-VL, OpenAI, Anthropic, Gemini)

## Installation

1.  **Cloner le dépôt**:
    ```bash
    git clone https://github.com/MohamedAzizChouaibi/GenowDatasetCleaner
    cd GenowDatasetCleaner
    ```

2.  **Créer un environnement virtuel** (recommandé):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sous Linux/macOS
    # venv\Scripts\activate   # Sous Windows
    ```

3.  **Installer les dépendances**:
    ```bash
    pip install -r requirements.txt
    ```

    **Note sur PyTorch et CLIP**: l'installation de `torch` peut nécessiter des étapes spécifiques selon votre configuration (CPU ou GPU). Consultez la documentation officielle de PyTorch et de CLIP en cas de problème. Le GPU (CUDA) est utilisé automatiquement s'il est disponible, sinon le CPU.

## Utilisation

### Interface graphique

```bash
python genowCleaner.py
```

1.  **Sélectionner le dataset**: cliquez sur "Browse…" pour choisir le dossier racine du dataset YOLO.
2.  **Régler les paramètres** (seuils de flou, luminosité, doublons, embedder, nombre de workers, clés API…).
3.  **Lancer l'analyse**: "Analyze dataset". L'application va:
    *   Collecter les images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
    *   Signaler les fichiers contenant `det` / `seg` (optionnel).
    *   Calculer les scores de qualité (flou, luminosité, information).
    *   Calculer les embeddings et regrouper les doublons.
4.  **Visualiser et nettoyer**:
    *   Les cartes de résultats affichent le nombre d'images par catégorie.
    *   "Review" pour inspecter et sélectionner les images à supprimer.
    *   "Auto-clean" pour conserver automatiquement la meilleure image de chaque cluster de doublons.
    *   La suppression retire aussi le fichier de label YOLO (`.txt`) associé. "Undo last delete" restaure la dernière suppression (quarantaine uniquement).

### Mode CLI (sans affichage)

```bash
python genowCleaner.py --cli /chemin/vers/dataset
```

Options principales:

*   `--report CHEMIN` — emplacement du rapport JSON (par défaut `~/.genow_cleaner/reports/`).
*   `--auto-clean-duplicates` — conserve la meilleure image par cluster, supprime les autres.
*   `--embedder CLE` — choix de l'embedder (ex. `clip:ViT-B/32`, `dinov2:base`…).
*   `--blur-threshold`, `--dark-threshold`, `--bright-threshold`, `--low-info-threshold`, `--duplicate-threshold` — seuils.
*   `--workers N` — nombre de threads.
*   `--no-phash` — désactive la pré-passe par hachage perceptuel.
*   `--use-trash` — utilise la corbeille système au lieu de la quarantaine.

## Structure du projet

```
GenowDatasetCleaner/
├── genowCleaner.py         # Application principale (GUI + CLI)
├── embedders.py            # Embedders locaux (CLIP, OpenCLIP, SigLIP, DINOv2)
├── api_embedders.py        # Embedders via API (Jina, Voyage, Qwen-VL, OpenAI, Claude, Gemini)
├── tests/                  # Tests
├── requirements.txt        # Dépendances Python
├── genow_logo.jpeg         # Logo de l'application
├── README.md               # Ce fichier
└── LICENSE                 # Licence du projet
```
