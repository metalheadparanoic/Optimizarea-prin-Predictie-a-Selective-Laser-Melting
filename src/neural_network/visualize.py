import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import cv2

# --- CONFIGURARE CAI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_OPT_DIR = os.path.join(PROJECT_ROOT, "docs", "optimization")
DOCS_RES_DIR = os.path.join(PROJECT_ROOT, "docs", "results")

# Asiguram folderele
os.makedirs(DOCS_OPT_DIR, exist_ok=True)
os.makedirs(DOCS_RES_DIR, exist_ok=True)

# Setare stil vizual
sns.set_theme(style="whitegrid")

def plot_metrics_evolution():
    """Genereaza graficul de evolutie a performantei pe etape."""
    print("[INFO] Generare metrics_evolution.png...")
    
    stages = ['Etapa 4\n(Untrained)', 'Etapa 5\n(Baseline)', 'Etapa 6\n(Optimizat)']
    accuracy = [50.07, 92.33, 99.67] 
    
    x = np.arange(len(stages))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, accuracy, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black', width=0.5)
    
    plt.ylabel('Acuratețe (%)', fontsize=12)
    plt.title('Evoluția Performanței Proiectului (Etapa 4 -> 6)', fontsize=14)
    plt.xticks(x, stages, fontsize=11)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    save_path = os.path.join(DOCS_RES_DIR, "metrics_evolution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Salvat in: {save_path}")

def plot_optimization_comparison():
    """Genereaza comparatia experimentelor (Accuracy si F1)."""
    # Verificam ambele nume posibile pentru CSV
    csv_path_1 = os.path.join(RESULTS_DIR, "optimization_experiments.csv")
    csv_path_2 = os.path.join(RESULTS_DIR, "optimization_experiments_full.csv")
    
    csv_path = csv_path_2 if os.path.exists(csv_path_2) else csv_path_1
    
    print(f"[INFO] Cautare date optimizare in: {csv_path} ...")

    if not os.path.exists(csv_path):
        print(f"[WARN] Fisierul CSV lipseste. Rulati optimize.py!")
        return

    try:
        df = pd.read_csv(csv_path)
        # Standardizare nume coloane (in caz ca difera intre versiuni)
        df.columns = [c.replace('Exp_ID', 'experiment_name').replace('Accuracy', 'val_accuracy') for c in df.columns]
        
        # Sortam
        df = df.sort_values(by='val_accuracy', ascending=True)

        # --- 1. Grafic Acuratețe ---
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='experiment_name', y='val_accuracy', data=df, palette='viridis', hue='experiment_name', legend=False)
        
        plt.title('Comparație Acuratețe Experimente (Etapa 6)', fontsize=14)
        plt.ylim(0.0, 1.1)
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Experiment')
        plt.xticks(rotation=15)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)

        save_path_acc = os.path.join(DOCS_OPT_DIR, "accuracy_comparison.png")
        plt.savefig(save_path_acc)
        plt.close()
        print(f"[OK] Salvat Accuracy: {save_path_acc}")

        # --- 2. Grafic F1-Score (ADĂUGAT ACUM) ---
        if 'F1_Score' in df.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='experiment_name', y='F1_Score', data=df, palette='magma', hue='experiment_name', legend=False)
            
            plt.title('Comparație F1-Score Experimente', fontsize=14)
            plt.ylim(0.0, 1.1)
            plt.ylabel('Validation F1-Score')
            plt.xlabel('Experiment')
            plt.xticks(rotation=15)
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', padding=3)

            save_path_f1 = os.path.join(DOCS_OPT_DIR, "f1_comparison.png")
            plt.savefig(save_path_f1)
            plt.close()
            print(f"[OK] Salvat F1: {save_path_f1}")
        else:
            print("[WARN] Coloana F1_Score lipseste din CSV.")

    except Exception as e:
        print(f"[EROARE] Nu s-au putut genera graficele de optimizare: {e}")
        import traceback
        traceback.print_exc()

def plot_learning_curves():
    """Genereaza curbele de invatare."""
    # NOTA: Acest fisier este generat de train.py. optimize.py NU il genereaza implicit.
    # Daca lipseste, vom crea unul fictiv pentru demonstratie sau vom da skip.
    csv_path = os.path.join(RESULTS_DIR, "training_history.csv")
    print(f"[INFO] Cautare learning curves in: {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} lipseste. (Rulati train.py pentru a-l genera).")
        return

    try:
        df = pd.read_csv(csv_path)
        epochs = range(1, len(df) + 1)

        plt.figure(figsize=(14, 6))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, df['accuracy'], 'b-o', label='Train Accuracy')
        plt.plot(epochs, df['val_accuracy'], 'r-o', label='Val Accuracy')
        plt.title('Acuratețe Antrenare vs Validare')
        plt.xlabel('Epoci')
        plt.ylabel('Acuratețe')
        plt.legend()
        plt.grid(True)

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, df['loss'], 'b-o', label='Train Loss')
        plt.plot(epochs, df['val_loss'], 'r-o', label='Val Loss')
        plt.title('Loss Antrenare vs Validare')
        plt.xlabel('Epoci')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(DOCS_RES_DIR, "learning_curves_final.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Salvat in: {save_path}")

    except Exception as e:
        print(f"[EROARE] Nu s-au putut genera curbele: {e}")

def plot_prediction_grid():
    """Genereaza grid predictii."""
    print("[INFO] Generare grid predictii...")
    
    # Cautam intai modelul manual (daca ai rulat train.py recent), apoi cel optimizat
    model_paths = [
        os.path.join(MODELS_DIR, "trained_model_manual_run.h5"),
        os.path.join(MODELS_DIR, "optimized_model.h5")
    ]
    
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        print("[WARN] Niciun model gasit pentru grid.")
        return

    print(f"[INFO] Folosim modelul: {os.path.basename(model_path)}")

    try:
        model = tf.keras.models.load_model(model_path)
        
        class_names = ["defect", "ok"]
        all_files = []
        
        for cls in class_names:
            cls_folder = os.path.join(DATA_TEST_DIR, cls)
            if os.path.exists(cls_folder):
                files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.endswith(('.png', '.jpg'))]
                all_files.extend([(f, cls) for f in files])
        
        if len(all_files) < 9:
            print("[WARN] Prea putine imagini in test pentru grid.")
            return
            
        selected_indices = np.random.choice(len(all_files), 9, replace=False)
        
        plt.figure(figsize=(10, 10))
        
        for i, idx in enumerate(selected_indices):
            filepath, true_label = all_files[idx]
            
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img_disp = cv2.resize(img, (200, 200))
            
            img_input = cv2.resize(img, (64, 64))
            img_input = np.expand_dims(img_input, axis=0)
            img_input = np.expand_dims(img_input, axis=-1)
            
            preds = model.predict(img_input, verbose=0)
            score = preds[0][0]
            
            pred_label = "ok" if score > 0.5 else "defect"
            conf = score if score > 0.5 else 1 - score
            
            plt.subplot(3, 3, i + 1)
            plt.imshow(img_disp, cmap='gray')
            plt.axis('off')
            
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {true_label.upper()}\nPred: {pred_label.upper()} ({conf*100:.1f}%)"
            plt.title(title, color=color, fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        save_path = os.path.join(DOCS_RES_DIR, "example_predictions.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Salvat in: {save_path}")

    except Exception as e:
        print(f"[EROARE] Grid predictii: {e}")

if __name__ == "__main__":
    plot_metrics_evolution()
    plot_optimization_comparison()
    plot_learning_curves()
    plot_prediction_grid()
    print("\n[DONE] Script finalizat.")