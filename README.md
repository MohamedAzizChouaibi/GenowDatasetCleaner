# GenowDatasetCleaner

**GenowDatasetCleaner** est une application de bureau simple et efficace, construite avec Python et Tkinter, conçue pour aider à nettoyer et à organiser les grands ensembles de données d'images. Il utilise des techniques avancées de vision par ordinateur, y compris le modèle **CLIP** (Contrastive Language-Image Pre-training) de OpenAI, pour identifier et signaler les images de mauvaise qualité ou redondantes.

## Fonctionnalités

*   **Détection de Doublons**: Utilise les embeddings CLIP pour calculer la similarité cosinus entre les images et identifier les doublons potentiels.
*   **Analyse de Qualité**:
    *   Détection des images **floues** (en utilisant la variance du Laplacien).
    *   Détection des images **sombres** ou **trop claires** (en analysant la luminosité moyenne).
    *   Détection des images à **faible information** (en analysant l'écart-type des pixels).
*   **Interface Utilisateur Graphique (GUI)**: Une interface conviviale basée sur Tkinter pour sélectionner le dossier du dataset, lancer l'analyse et visualiser les résultats.
*   **Nettoyage Automatisé**: Option de suppression des images identifiées comme problématiques (doublons, floues, etc.) et de leurs fichiers XML associés.

## Dépendances

Ce projet nécessite les bibliothèques Python suivantes. Elles sont listées dans le fichier `requirements.txt`.

*   `pandas`
*   `numpy`
*   `opencv-python` (cv2)
*   `Pillow` (PIL)
*   `matplotlib`
*   `scikit-learn`
*   `torch`
*   `clip`

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

    **Note sur PyTorch et CLIP**: L'installation de `torch` et `clip` peut nécessiter des étapes spécifiques en fonction de votre configuration (CPU ou GPU). Veuillez consulter la documentation officielle de PyTorch et du modèle CLIP si vous rencontrez des problèmes.

## Utilisation

1.  **Lancer l'application**:
    ```bash
    python genowCleaner.py
    ```

2.  **Sélectionner le Dataset**: Cliquez sur "Parcourir" pour choisir le dossier racine de votre dataset d'images.
3.  **Démarrer l'Analyse**: Cliquez sur "Démarrer l'Analyse". L'application va:
    *   Collecter les images (`.jpg`, `.jpeg`, `.png`).
    *   Supprimer les images contenant "det" ou "seg" dans leur nom de fichier (et leurs XML associés).
    *   Calculer les scores de qualité (flou, luminosité, information).
    *   Calculer les embeddings CLIP et identifier les doublons.
4.  **Visualiser et Nettoyer**:
    *   Le résumé affichera le nombre d'images problématiques trouvées.
    *   Utilisez les boutons de visualisation pour inspecter les images signalées.
    *   Cliquez sur le bouton rouge "Terminer le Nettoyage (Supprimer les Images)" pour supprimer définitivement les images identifiées.

## Structure du Projet

```
GenowDatasetCleaner/
├── genowCleaner.py         # Le script principal de l'application
├── requirements.txt        # Liste des dépendances Python
├── README.md               # Ce fichier
├── .gitignore              # Fichiers et dossiers à ignorer par Git
└── LICENSE                 # Licence du projet (à ajouter)
```
