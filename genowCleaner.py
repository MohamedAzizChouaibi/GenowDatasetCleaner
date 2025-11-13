import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip

# --- Configuration et Constantes ---
# Seuil de détection des doublons
DUPLICATE_THRESHOLD = 0.96
ICON_PATH = "/home/shuayb/Downloads/genow.ico"
# Seuils d'analyse d'image
BLUR_THRESHOLD = 60
DARK_THRESHOLD = 10
BRIGHT_THRESHOLD = 230
LOW_INFO_THRESHOLD = 12

# Définir le périphérique pour CLIP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Fonctions d'Analyse d'Image ---

def get_brightness_and_std(image_path):
    """Calcule la luminosité moyenne et l'écart-type d'une image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        # Convertir en niveaux de gris pour l'écart-type et la luminosité
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Luminosité moyenne (moyenne des pixels)
        brightness_mean = np.mean(gray)
        
        # Écart-type (pour la détection de faible information)
        std_dev = np.std(gray)
        
        return brightness_mean, std_dev
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return None, None

def get_blur_score(image_path):
    """Calcule le score de flou (variance du Laplacien)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Variance du Laplacien
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        print(f"Erreur lors du calcul du flou pour {image_path}: {e}")
        return None

# --- Fonctions de Visualisation ---

def show_image(path, title="", size=(6,6)):
    """Affiche une seule image."""
    try:
        img = Image.open(path).convert("RGB")
        plt.figure(figsize=size)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()
    except Exception as e:
        messagebox.showerror("Erreur de Visualisation", f"Impossible d'afficher l'image {path}: {e}")

def show_duplicates_visualization(duplicates_df, dataset_path):
    """Affiche les paires de doublons côte à côte."""
    if duplicates_df.empty:
        messagebox.showinfo("Doublons", "Aucun doublon trouvé à visualiser.")
        return

    # Prendre les 5 premières paires pour la visualisation
    for index, row in duplicates_df.head(5).iterrows():
        img1_path = os.path.join(dataset_path, row['img1'])
        img2_path = os.path.join(dataset_path, row['img2'])
        similarity = row['similarity']

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Doublons - Similitude: {similarity:.4f}", fontsize=16)

            axes[0].imshow(img1)
            axes[0].set_title(f"Image 1: {os.path.basename(img1_path)}")
            axes[0].axis("off")

            axes[1].imshow(img2)
            axes[1].set_title(f"Image 2 (à supprimer): {os.path.basename(img2_path)}")
            axes[1].axis("off")

            plt.show()
        except Exception as e:
            messagebox.showerror("Erreur de Visualisation", f"Impossible d'afficher la paire {img1_path} et {img2_path}: {e}")

def show_single_image_visualization(df, dataset_path, column_name, title_prefix):
    """Affiche les images problématiques (floues, sombres, claires, faible info)."""
    if df.empty:
        messagebox.showinfo(title_prefix, f"Aucune image {title_prefix.lower()} trouvée à visualiser.")
        return

    # Prendre les 5 premières images pour la visualisation
    for index, row in df.head(5).iterrows():
        img_path = os.path.join(dataset_path, row['image'])
        score = row[column_name]
        
        show_image(img_path, title=f"{title_prefix}: {score:.2f}")

# --- Classe Principale de l'Application ---

class DatasetCleanerApp:
    def __init__(self, master):
        self.master = master

        master.title("GenowDatasetCleaner")

        self.dataset_path = tk.StringVar()
        self.summary_text = None
        self.progress_bar = None
        self.start_button = None
        self.cleaning_button = None
        self.visualization_frame = None
        # --- Background Image ---


        self.results = {
            'duplicates': pd.DataFrame(columns=['img1', 'img2', 'similarity']),
            'blurry': pd.DataFrame(columns=['image', 'blur_score']),
            'dark': pd.DataFrame(columns=['image', 'brightness_mean']),
            'bright': pd.DataFrame(columns=['image', 'brightness_mean']),
            'low_information': pd.DataFrame(columns=['image', 'std_dev']),
        }
        
        self.create_widgets()
        self.load_clip_model()

    def load_clip_model(self):
        """Charge le modèle CLIP."""
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
            self.model.eval()
            self.update_summary(f"Modèle CLIP chargé sur {DEVICE}.")
        except Exception as e:
            messagebox.showerror("Erreur CLIP", f"Impossible de charger le modèle CLIP: {e}")
            self.update_summary("Erreur: Impossible de charger le modèle CLIP.")
            self.start_button.config(state=tk.DISABLED)

    def create_widgets(self):
        """Crée tous les éléments de l'interface graphique."""
        
        # Frame principale pour l'entrée
        input_frame = ttk.Frame(self.master, padding="10")
        input_frame.pack(fill='x')

        # Label et Champ de saisie du chemin
        ttk.Label(input_frame, text="Entrez le chemin du dataset:").pack(side='left', padx=5, pady=5)
        self.path_entry = ttk.Entry(input_frame, textvariable=self.dataset_path, width=50)
        self.path_entry.pack(side='left', fill='x', expand=True, padx=5, pady=5)
        
        # Bouton Parcourir
        ttk.Button(input_frame, text="Parcourir", command=self.browse_path).pack(side='left', padx=5, pady=5)

        # Bouton Démarrer l'Analyse
        self.start_button = ttk.Button(self.master, text="Démarrer l'Analyse", command=self.start_analysis)
        self.start_button.pack(pady=10)

        # Barre de Progression
        self.progress_bar = ttk.Progressbar(self.master, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Zone de Résumé
        ttk.Label(self.master, text="Résumé de l'Analyse:").pack(pady=(10, 0))
        self.summary_text = tk.Text(self.master, height=10, width=60, state=tk.DISABLED)
        self.summary_text.pack(padx=10, pady=(0, 10))
        
        # Frame pour les boutons post-analyse
        self.post_analysis_frame = ttk.Frame(self.master, padding="10")
        self.post_analysis_frame.pack(fill='x')
        
        # Bouton de Nettoyage (initialement caché)
        self.cleaning_button = ttk.Button(self.post_analysis_frame, text="🔴 Terminer le Nettoyage (Supprimer les Images)", command=self.start_cleaning, style='Red.TButton')
        self.cleaning_button.pack(pady=10)
        self.cleaning_button.pack_forget() # Cacher initialement
        
        # Style pour le bouton de nettoyage
        style = ttk.Style()
        style.configure('Red.TButton', foreground='red', font=('Helvetica', 10, 'bold'))

        # Frame pour les boutons de visualisation (initialement cachée)
        self.visualization_frame = ttk.Frame(self.post_analysis_frame)
        self.visualization_frame.pack(pady=10)
        self.visualization_frame.pack_forget() # Cacher initialement

    def browse_path(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier."""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.dataset_path.set(folder_path)
            self.update_summary(f"Chemin du dataset sélectionné: {folder_path}")

    def update_summary(self, message):
        """Met à jour la zone de résumé."""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.insert(tk.END, message + "\n")
        self.summary_text.see(tk.END)
        self.summary_text.config(state=tk.DISABLED)
        self.master.update_idletasks() # Assure la réactivité de l'interface

    def update_progress(self, value, max_value, message=""):
        """Met à jour la barre de progression."""
        if max_value > 0:
            self.progress_bar['value'] = (value / max_value) * 100
        else:
            self.progress_bar['value'] = 0
        self.master.update_idletasks()
        if message:
            self.update_summary(message)

    def start_analysis(self):
        """Lance le processus d'analyse complet."""
        path = self.dataset_path.get()
        if not os.path.isdir(path):
            messagebox.showerror("Erreur de Chemin", "Veuillez entrer un chemin de dossier valide.")
            return

        # Réinitialiser l'état
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.config(state=tk.DISABLED)
        self.cleaning_button.pack_forget()
        self.visualization_frame.pack_forget()
        self.start_button.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        self.results = {k: pd.DataFrame(columns=v.columns) for k, v in self.results.items()} # Réinitialiser les DataFrames

        self.update_summary("Début de l'analyse...")
        
        # 1. Collecte des fichiers
        image_files = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(path, '**', '*.jpeg'), recursive=True) + \
                      glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
        # --- REMOVE all images containing "det" or "seg" in their filenames ---
        removed_images = []
        filtered_images = []

        for img in image_files:
            filename = os.path.basename(img).lower()
            if "det" in filename or "seg" in filename:
                # Delete the image + its XML
                base, ext = os.path.splitext(img)
                xml_path = base + ".xml"

                try:
                    os.remove(img)
                    removed_images.append(img)
                except:
                    pass

                if os.path.exists(xml_path):
                    try:
                        os.remove(xml_path)
                        removed_images.append(xml_path)
                    except:
                        pass

            else:
                filtered_images.append(img)

        image_files = filtered_images
        self.update_summary(f"{len(removed_images)} fichiers 'det' ou 'seg' supprimés avant analyse.")

        if not image_files:
            self.update_summary("Aucun fichier image trouvé. Fin de l'analyse.")
            self.start_button.config(state=tk.NORMAL)
            return

        self.update_summary(f"Trouvé {len(image_files)} images. Début de l'analyse CV2/Numpy.")
        
        # 2. Analyse CV2/Numpy (Flou, Luminosité, Faible Info)
        self.cv2_analysis(image_files, path)
        
        # 3. Analyse CLIP (Doublons)
        self.update_summary("Début de l'analyse des doublons (CLIP/Torch)...")
        self.clip_analysis(image_files, path)
        
        # 4. Affichage des résultats et préparation de l'interface
        self.finalize_analysis(path)

    def cv2_analysis(self, image_files, dataset_path):
        """Effectue l'analyse CV2/Numpy pour le flou, la luminosité et l'info."""
        total_images = len(image_files)
        
        for i, full_path in enumerate(image_files):
            relative_path = os.path.relpath(full_path, dataset_path)
            
            # Mise à jour de la progression
            self.update_progress(i + 1, total_images, f"Analyse CV2: {relative_path}")
            
            # 1. Flou
            blur_score = get_blur_score(full_path)
            if blur_score is not None and blur_score < BLUR_THRESHOLD:
                self.results['blurry'].loc[len(self.results['blurry'])] = [relative_path, blur_score]
            
            # 2. Luminosité et Écart-type
            brightness_mean, std_dev = get_brightness_and_std(full_path)
            if brightness_mean is not None:
                if brightness_mean < DARK_THRESHOLD:
                    self.results['dark'].loc[len(self.results['dark'])] = [relative_path, brightness_mean]
                elif brightness_mean > BRIGHT_THRESHOLD:
                    self.results['bright'].loc[len(self.results['bright'])] = [relative_path, brightness_mean]
            
            if std_dev is not None and std_dev < LOW_INFO_THRESHOLD:
                self.results['low_information'].loc[len(self.results['low_information'])] = [relative_path, std_dev]

    def clip_analysis(self, image_files, dataset_path):
        """Effectue l'analyse CLIP pour la détection de doublons."""
        total_images = len(image_files)
        
        # 1. Calcul des embeddings
        embeddings = []
        paths = []
        
        for i, full_path in enumerate(image_files):
            relative_path = os.path.relpath(full_path, dataset_path)
            self.update_progress(i + 1, total_images, f"Calcul des embeddings: {relative_path}")
            
            try:
                image = self.preprocess(Image.open(full_path)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                embeddings.append(image_features.cpu().numpy().flatten())
                paths.append(relative_path)
            except Exception as e:
                self.update_summary(f"Avertissement: Impossible de calculer l'embedding pour {relative_path}: {e}")
                continue

        if not embeddings:
            self.update_summary("Aucun embedding calculé. Impossible de vérifier les doublons.")
            return

        embeddings_matrix = np.array(embeddings)
        
        # 2. Calcul de la matrice de similarité cosinus
        self.update_summary("Calcul de la matrice de similarité cosinus...")
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # 3. Extraction des doublons
        duplicates = []
        # On ne vérifie que la moitié supérieure de la matrice (i < j) pour éviter les paires (A, B) et (B, A) et (A, A)
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                similarity = similarity_matrix[i, j]
                if similarity > DUPLICATE_THRESHOLD:
                    duplicates.append({
                        'img1': paths[i],
                        'img2': paths[j],
                        'similarity': similarity
                    })
        
        self.results['duplicates'] = pd.DataFrame(duplicates)
        self.update_progress(total_images, total_images, "Analyse des doublons terminée.")

    def finalize_analysis(self, dataset_path):
        """Termine l'analyse, affiche le résumé et les boutons."""
        
        # 1. Sauvegarde des CSV
        output_dir = os.path.join(dataset_path, "cleaning_results")
        os.makedirs(output_dir, exist_ok=True)
        
        summary_lines = ["--- RÉSULTATS DE L'ANALYSE ---"]
        
        for key, df in self.results.items():
            csv_filename = f"{key}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Renommer les colonnes pour correspondre aux exigences
            if key == 'blurry':
                df.rename(columns={'blur_score': 'blur_score'}, inplace=True)
            elif key == 'dark' or key == 'bright':
                df.rename(columns={'brightness_mean': 'brightness_mean'}, inplace=True)
            elif key == 'low_information':
                df.rename(columns={'std_dev': 'std_dev'}, inplace=True)
            
            df.to_csv(csv_path, index=False)
            summary_lines.append(f"  - {len(df)} images trouvées dans la catégorie '{key}'.")
            summary_lines.append(f"    (Détails sauvegardés dans {csv_filename})")

        self.update_summary("\n".join(summary_lines))
        self.update_summary(f"Fichiers CSV de résultats sauvegardés dans: {output_dir}")
        
        # 2. Affichage des boutons post-analyse
        self.setup_post_analysis_buttons()
        self.start_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 100

    def setup_post_analysis_buttons(self):
        """Crée et affiche les boutons de nettoyage et de visualisation."""
        
        # Bouton de Nettoyage
        self.cleaning_button.pack(pady=10)
        
        # Boutons de Visualisation
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
            
        buttons_data = [
            ("Doublons", 'duplicates', 'img1', 'Similitude', show_duplicates_visualization),
            ("Images Floues", 'blurry', 'blur_score', 'Score de Flou', show_single_image_visualization),
            ("Images Sombres", 'dark', 'brightness_mean', 'Luminosité Moyenne', show_single_image_visualization),
            ("Images Claires", 'bright', 'brightness_mean', 'Luminosité Moyenne', show_single_image_visualization),
            ("Images Faible Info", 'low_information', 'std_dev', 'Écart-type', show_single_image_visualization),
        ]
        
        for text, key, col_name, title_prefix, func in buttons_data:
            # Ajouter le nombre d'images trouvées au texte du bouton
            count = len(self.results[key])
            button_text = f"Afficher {text} ({count})"
            
            if key == 'duplicates':
                # Cas spécial pour les doublons
                command = lambda k=key, f=func: f(self.results[k], self.dataset_path.get())
            else:
                # Cas général pour les autres
                command = lambda k=key, c=col_name, t=title_prefix, f=show_single_image_visualization: f(self.results[k], self.dataset_path.get(), c, t)
                
            ttk.Button(self.visualization_frame, text=button_text, command=command).pack(side='left', padx=5, pady=5)

        self.visualization_frame.pack(pady=10)

    def delete_file_and_xml(self, relative_path, dataset_path):
        """Supprime un fichier image et son fichier .xml correspondant."""
        full_path = os.path.join(dataset_path, relative_path)
        
        # Chemin du fichier XML
        base, ext = os.path.splitext(full_path)
        xml_path = base + ".xml"
        
        deleted_files = []
        
        # 1. Suppression de l'image
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                deleted_files.append(full_path)
            except Exception as e:
                self.update_summary(f"Erreur: Impossible de supprimer l'image {full_path}: {e}")
        
        # 2. Suppression du XML
        if os.path.exists(xml_path):
            try:
                os.remove(xml_path)
                deleted_files.append(xml_path)
            except Exception as e:
                self.update_summary(f"Erreur: Impossible de supprimer le XML {xml_path}: {e}")
                
        return deleted_files

    def start_cleaning(self):
        """Lance le processus de nettoyage (suppression des fichiers)."""
        
        if not messagebox.askyesno("Confirmation de Nettoyage", 
                                   "Êtes-vous sûr de vouloir SUPPRIMER DÉFINITIVEMENT les images et leurs fichiers XML associés ? Cette action est irréversible."):
            return

        dataset_path = self.dataset_path.get()
        self.update_summary("\n--- DÉBUT DU NETTOYAGE ---")
        self.cleaning_button.config(state=tk.DISABLED)
        
        total_deleted_count = 0
        
        # 1. Images Claires, Sombres, Floues, Faible Info
        problem_categories = [
            ('bright', 'Images Claires'), 
            ('dark', 'Images Sombres'), 
            ('blurry', 'Images Floues'), 
            ('low_information', 'Images Faible Info')
        ]
        
        for key, name in problem_categories:
            df = self.results[key]
            self.update_summary(f"Nettoyage des {name} ({len(df)} images)...")
            
            for index, row in df.iterrows():
                deleted = self.delete_file_and_xml(row['image'], dataset_path)
                total_deleted_count += len(deleted)
                self.update_progress(index + 1, len(df), f"Supprimé: {row['image']}")
        
        # 2. Doublons (Supprimer img2)
        duplicates_df = self.results['duplicates']
        self.update_summary(f"Nettoyage des Doublons ({len(duplicates_df)} paires)...")
        
        for index, row in duplicates_df.iterrows():
            # Supprimer img2 et son XML
            deleted = self.delete_file_and_xml(row['img2'], dataset_path)
            total_deleted_count += len(deleted)
            self.update_progress(index + 1, len(duplicates_df), f"Supprimé doublon (img2): {row['img2']}")

        self.update_summary(f"\nNettoyage terminé. Total de fichiers supprimés (images + xml): {total_deleted_count}")
        
        # Affichage du popup final
        messagebox.showinfo("Nettoyage Terminé", "Nettoyage terminé avec succès!")
        
        # Réinitialiser l'interface pour une nouvelle analyse
        self.cleaning_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
        self.visualization_frame.pack_forget()
        self.update_summary("Veuillez redémarrer l'analyse pour mettre à jour les résultats.")


if __name__ == "__main__":
    # Vérifier les dépendances critiques (torch et clip)
    try:
        import torch
        import clip
    except ImportError:
        print("Erreur: Les dépendances 'torch' et 'clip' ne sont pas installées.")
        print("Veuillez exécuter: pip install torch torchvision torchaudio")
        print("Puis: pip install ftfy regex tqdm")
        print("Puis: pip install git+https://github.com/openai/CLIP.git")
        exit()
        
    # Vérifier les autres dépendances
    try:
        import cv2
        import pandas as pd
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        print(f"Erreur: Une dépendance critique est manquante: {e}")
        print("Veuillez exécuter: pip install opencv-python pandas numpy Pillow matplotlib scikit-learn")
        exit()

    root = tk.Tk()
    app = DatasetCleanerApp(root)
    root.mainloop()
