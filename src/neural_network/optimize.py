import os
import sys
import csv
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

# --- CONFIGURARE CAI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.neural_network.model import create_model

# Cai foldere
DATA_DIR = os.path.join(project_root, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
RESULTS_DIR = os.path.join(project_root, "results")
MODELS_DIR = os.path.join(project_root, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (64, 64)

def run_optimization():
    print("\n[INFO] Start Experimente Optimizare (Conform Tabel)...")

    # --- DEFINIRE CELE 6 EXPERIMENTE ---
    experiments = [
        # Baseline
        {"id": "Baseline", "lr": 0.001,  "batch": 32, "dropout": 0.5, "extra": False, "noise": False, "desc": "Referinta"},
        
        # Exp 1: LR Mic
        {"id": "Exp_1",    "lr": 0.0001, "batch": 32, "dropout": 0.5, "extra": False, "noise": False, "desc": "Convergenta fina"},
        
        # Exp 2: Batch Mare
        {"id": "Exp_2",    "lr": 0.001,  "batch": 64, "dropout": 0.5, "extra": False, "noise": False, "desc": "Viteza mare"},
        
        # Exp 3: Extra Layer
        {"id": "Exp_3",    "lr": 0.001,  "batch": 32, "dropout": 0.5, "extra": True,  "noise": False, "desc": "Capacitate crescuta"},
        
        # Exp 4: Dropout Mare
        {"id": "Exp_4",    "lr": 0.001,  "batch": 32, "dropout": 0.7, "extra": False, "noise": False, "desc": "Regularizare agresiva"},
        
        # Exp 5: Augmentare Gaussian Noise (BEST)
        {"id": "Exp_5",    "lr": 0.001,  "batch": 32, "dropout": 0.5, "extra": False, "noise": True,  "desc": "Augmentare robusta"},
    ]

    csv_file = os.path.join(RESULTS_DIR, "optimization_experiments.csv")
    history_file = os.path.join(RESULTS_DIR, "training_history.csv")
    
    best_acc = 0.0
    best_exp_id = ""

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header similar cu tabelul tau
        writer.writerow(["Exp_ID", "Accuracy", "F1_Score", "Timp_Antrenare", "Observatii"])

        for exp in experiments:
            print(f"\n---> Rulare {exp['id']}: {exp['desc']}...")

            # 1. Incarcare Date
            try:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    TRAIN_DIR, image_size=IMG_SIZE, batch_size=exp['batch'],
                    color_mode='grayscale', label_mode='binary', shuffle=True, verbose=0
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    VAL_DIR, image_size=IMG_SIZE, batch_size=exp['batch'],
                    color_mode='grayscale', label_mode='binary', shuffle=False, verbose=0
                )
            except Exception as e:
                print(f"[EROARE] Date lipsa: {e}")
                return

            train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            # 2. Configurare Model
            model = create_model(
                dropout_rate=exp['dropout'],
                extra_layer=exp['extra'],
                use_noise=exp['noise']
            )
            
            # Compilam cu metrici pentru F1 (Precision, Recall)
            model.compile(optimizer=optimizers.Adam(learning_rate=exp['lr']),
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            
            # 3. Antrenare
            start_time = time.time()
            history = model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=0)
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            # 4. Calcule Finale
            final_acc = history.history['val_accuracy'][-1]
            
            # Calculam F1 manual din Precision si Recall
            # TensorFlow pune uneori sufixe _1, _2 la chei, asa ca le cautam dinamic
            keys = history.history.keys()
            prec_key = next((k for k in keys if 'precision' in k and 'val' in k), None)
            rec_key = next((k for k in keys if 'recall' in k and 'val' in k), None)
            
            if prec_key and rec_key:
                p = history.history[prec_key][-1]
                r = history.history[rec_key][-1]
                if (p + r) > 0:
                    f1 = 2 * (p * r) / (p + r)
                else:
                    f1 = 0.0
            else:
                f1 = final_acc # Fallback daca ceva merge prost

            print(f"     [REZULTAT] Acc: {final_acc:.4f} | F1: {f1:.4f} | Time: {duration}s")

            # 5. Salvare Best Model
            if final_acc > best_acc:
                best_acc = final_acc
                best_exp_id = exp['id']
                
                model.save(os.path.join(MODELS_DIR, "optimized_model.h5"))
                
                # Salvam istoricul pentru grafice
                hist_df = pd.DataFrame(history.history)
                hist_df.to_csv(history_file, index=False)
                print(f"     >>> NEW RECORD! Model salvat.")

            # 6. Scriere CSV
            writer.writerow([exp['id'], round(final_acc, 4), round(f1, 4), f"{duration}s", exp['desc']])

    print(f"\n[FINAL] Cel mai bun model: {best_exp_id}")

if __name__ == "__main__":
    run_optimization()