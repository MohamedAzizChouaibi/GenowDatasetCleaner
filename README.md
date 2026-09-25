# GenowDatasetCleaner

**GenowDatasetCleaner** est une application de bureau (Python + Tkinter) conçue pour nettoyer et organiser des datasets d'images : **n'importe quel dossier d'images** ou un dataset au format **YOLOv11 / YOLOv12** (Ultralytics). Elle détecte les images de mauvaise qualité (floues, sur/sous-exposées, peu informatives) et les doublons à l'aide d'embeddings visuels (**CLIP** par défaut, plusieurs autres VLM disponibles), puis permet de les supprimer (avec leurs fichiers de labels YOLO `.txt` associés dans le cas d'un dataset YOLO).

Le format YOLO **n'est pas obligatoire** :

*   **Dataset YOLO** — un dossier contenant soit un `data.yaml` avec les clés `train:` et `names:`/`nc:`, soit des répertoires parallèles `images/` et `labels/`. La suppression d'une image retire aussi son label `.txt`, et les splits `train`/`val`/`test` sont pris en compte pour l'auto-clean.
*   **Dossier d'images quelconque** — tout autre dossier est analysé comme une simple collection d'images (parcours récursif). Seules les images sont supprimées ; aucun autre fichier n'est touché.

## Fonctionnalités

*   **Détection de doublons**: pré-passe par hachage perceptuel (pHash) pour les doublons exacts, puis clustering par similarité cosinus sur les embeddings VLM. Chaque image d'un cluster ressemble à l'image conservée (pas de chaînage A≈B≈C).
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

1.  **Sélectionner le dataset**: cliquez sur "Browse…" pour choisir le dossier racine du dataset (YOLO ou simple dossier d'images).
2.  **Régler les paramètres** (seuils de flou, luminosité, doublons, embedder, nombre de workers, clés API…). Changer d'embedder le recharge immédiatement en arrière-plan ; en cas d'échec (clé API manquante, dépendance absente), l'embedder précédent est restauré.
3.  **Lancer l'analyse**: "Analyze dataset". L'application va:
    *   Collecter les images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
    *   Signaler les fichiers contenant `det` / `seg` (optionnel).
    *   Calculer les scores de qualité (flou, luminosité, information).
    *   Calculer les embeddings et regrouper les doublons.
4.  **Visualiser et nettoyer**:
    *   Les cartes de résultats affichent le nombre d'images par catégorie.
    *   "Review" pour inspecter et sélectionner les images à supprimer (aucune n'est cochée par défaut). Les fenêtres sont paginées (50 images ou 15 clusters par page). Cliquer sur une miniature l'affiche en grand (←/→ pour naviguer, Suppr pour cocher/décocher, Échap pour fermer).
    *   Raccourcis clavier des fenêtres de revue : Échap fermer · ←/→ page précédente/suivante · Suppr supprimer la sélection · Ctrl+A / Ctrl+D tout sélectionner / rien (pour les doublons : inclure / ignorer tous les clusters).
    *   Dans la revue des doublons, cochez "Keep this" sur **une ou plusieurs** images par cluster (la première est cochée par défaut) ; seules les images non cochées sont supprimées. Au moins une image doit rester cochée — "Skip this cluster" conserve tout le cluster.
    *   "Clean all problems" supprime en une fois toutes les images signalées (toutes catégories) et toutes les copies en double sauf une par cluster, après confirmation détaillée.
    *   "Auto-clean" pour conserver automatiquement une image par cluster de doublons : la copie en `test`/`val` est prioritaire sur `train` (évite de vider les splits d'évaluation), puis la meilleure qualité. Les clusters répartis sur plusieurs splits (fuite train/val/test) sont signalés.
    *   Pour un dataset YOLO, la suppression retire aussi le fichier de label (`.txt`) associé. "Undo last delete" restaure la dernière suppression (quarantaine uniquement).
    *   "Open reports" ouvre le dossier des rapports JSON (`~/.genow_cleaner/reports/`).

### Mode CLI (sans affichage)

```bash
python genowCleaner.py --cli /chemin/vers/dataset
```

Options principales:

*   `--report CHEMIN` — emplacement du rapport JSON (par défaut `~/.genow_cleaner/reports/`).
*   `--auto-clean-duplicates` — conserve une image par cluster (copie `test`/`val` d'abord, puis meilleure qualité), supprime les autres.
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
