import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import cv2

# --- CONFIGURARE CAI ---
# Determinam calea absoluta catre radacina proiectului
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
    """Genereaza graficul de evolutie a performantei pe etape (din generate_final_plots.py)."""
    print("[INFO] Generare metrics_evolution.png...")
    
    stages = ['Etapa 4\n(Untrained)', 'Etapa 5\n(Baseline)', 'Etapa 6\n(Optimizat)']
    # Valori istorice reale din proiectul tau
    accuracy = [50.07, 92.33, 99.67] 
    
    x = np.arange(len(stages))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, accuracy, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black', width=0.5)
    
    plt.ylabel('Acuratețe (%)', fontsize=12)
    plt.title('Evoluția Performanței Proiectului (Etapa 4 -> 6)', fontsize=14)
    plt.xticks(x, stages, fontsize=11)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adaugare etichete
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    save_path = os.path.join(DOCS_RES_DIR, "metrics_evolution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Salvat in: {save_path}")

def plot_optimization_comparison():
    """Genereaza comparatia experimentelor (din generate_plots.py)."""
    csv_path = os.path.join(RESULTS_DIR, "optimization_experiments.csv")
    print(f"[INFO] Generare grafice optimizare din {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"[WARN] Fisierul {csv_path} lipseste. Sar peste acest pas.")
        return

    try:
        df = pd.read_csv(csv_path)
        # Sortam pentru aspect placut
        df = df.sort_values(by='val_accuracy', ascending=True)

        # 1. Grafic Acuratețe
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='experiment_name', y='val_accuracy', data=df, palette='viridis', hue='experiment_name', legend=False)
        
        plt.title('Comparație Acuratețe Experimente (Etapa 6)', fontsize=14)
        plt.ylim(0.0, 1.1) # Scala 0-1
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Experiment')
        plt.xticks(rotation=15)
        
        # Etichete pe bare
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)

        save_path = os.path.join(DOCS_OPT_DIR, "accuracy_comparison.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Salvat in: {save_path}")

    except Exception as e:
        print(f"[EROARE] Nu s-au putut genera graficele de optimizare: {e}")

def plot_learning_curves():
    """Genereaza curbele de invatare pentru modelul final."""
    csv_path = os.path.join(RESULTS_DIR, "training_history.csv")
    print(f"[INFO] Generare learning curves din {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} lipseste.")
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
    """Genereaza un grid de exemple (Real vs Predis) folosind modelul optimizat."""
    print("[INFO] Generare grid predictii (example_predictions.png)...")
    model_path = os.path.join(MODELS_DIR, "optimized_model.h5")
    
    if not os.path.exists(model_path):
        print("[WARN] Modelul optimizat lipseste. Sar peste grid.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        
        # Selectam cateva imagini random din test
        images = []
        labels = []
        preds = []
        
        # Cautam in folderele test
        class_names = ["defect", "ok"]
        all_files = []
        
        for cls in class_names:
            cls_folder = os.path.join(DATA_TEST_DIR, cls)
            if os.path.exists(cls_folder):
                files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.endswith(('.png', '.jpg'))]
                # Adaugam tuplu (cale, label_real)
                all_files.extend([(f, cls) for f in files])
        
        if len(all_files) < 9:
            print("[WARN] Prea putine imagini pentru grid.")
            return
            
        # Alegem 9 random
        selected = np.random.choice(len(all_files), 9, replace=False)
        
        plt.figure(figsize=(10, 10))
        
        for i, idx in enumerate(selected):
            filepath, true_label = all_files[idx]
            
            # Citire si preprocesare
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img_disp = cv2.resize(img, (200, 200)) # Pentru afisare clara
            
            img_input = cv2.resize(img, (64, 64))
            img_input = np.expand_dims(img_input, axis=0) # Batch
            img_input = np.expand_dims(img_input, axis=-1) # Channel
            
            # Predictie
            score = model.predict(img_input, verbose=0)[0][0]
            pred_label = "ok" if score > 0.5 else "defect"
            conf = score if score > 0.5 else 1 - score
            
            # Plot
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
    # Rulam toate functiile de generare
    plot_metrics_evolution()
    plot_optimization_comparison()
    plot_learning_curves()
    plot_prediction_grid()
    print("\n[DONE] Vizualizare completa finalizata.")