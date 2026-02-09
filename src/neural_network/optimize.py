import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import csv
import time
import numpy as np
import sys

# Setup cai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- CONFIGURARE ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (64, 64)

def build_model(filters_start=32, dropout=0.5, lr=0.001, extra_layer=False):
    model = models.Sequential()
    
    # Fix Warning: Folosim Input layer explicit
    model.add(layers.Input(shape=(64, 64, 1)))
    model.add(layers.Rescaling(1./255))
    model.add(layers.RandomFlip("horizontal_and_vertical"))
    
    # Strat 1
    model.add(layers.Conv2D(filters_start, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Strat 2
    model.add(layers.Conv2D(filters_start * 2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Strat 3
    model.add(layers.Conv2D(filters_start * 4, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # EXPERIMENT 3: Extra Hidden Layer
    model.add(layers.Flatten())
    
    if extra_layer:
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(dropout))
    else:
        model.add(layers.Dense(filters_start * 2, activation='relu'))
        model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # FIX ERROARE KEYERROR: Fortam numele metricilor
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(name='precision'), 
                           tf.keras.metrics.Recall(name='recall')])
    return model

def main():
    print("\n Start Experimente Optimizare...")

    # Definire Experimente (Baseline + 5 Variatii)
    experiments = [
        {"id": "Baseline", "mod": "Config Etapa 5", "lr": 0.001, "batch": 32, "extra": False, "drop": 0.5, "aug": False},
        {"id": "Exp 1", "mod": "LR 0.001 -> 0.0001", "lr": 0.0001, "batch": 32, "extra": False, "drop": 0.5, "aug": False},
        {"id": "Exp 2", "mod": "Batch 32 -> 64", "lr": 0.001, "batch": 64, "extra": False, "drop": 0.5, "aug": False},
        {"id": "Exp 3", "mod": "+1 Hidden Layer (128)", "lr": 0.001, "batch": 32, "extra": True, "drop": 0.5, "aug": False},
        {"id": "Exp 4", "mod": "Dropout 0.5 -> 0.7", "lr": 0.001, "batch": 32, "extra": False, "drop": 0.7, "aug": False},
        {"id": "Exp 5", "mod": "Extra Augmentare (Gaussian)", "lr": 0.001, "batch": 32, "extra": False, "drop": 0.5, "aug": True},
    ]

    csv_file = os.path.join(RESULTS_DIR, "optimization_experiments_full.csv")
    
    # Deschidem fisierul CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Exp_ID", "Modificare", "Accuracy", "F1_Score", "Timp_Antrenare", "Observatii"])

        for exp in experiments:
            print(f"\n Rulare {exp['id']}: {exp['mod']}...")
            
            # Reincarcare Dataset pentru a aplica batch size diferit
            train_ds = tf.keras.utils.image_dataset_from_directory(
                TRAIN_DIR, image_size=IMG_SIZE, batch_size=exp['batch'],
                color_mode='grayscale', label_mode='binary', shuffle=True, verbose=0
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                VAL_DIR, image_size=IMG_SIZE, batch_size=exp['batch'],
                color_mode='grayscale', label_mode='binary', verbose=0
            )

            # Augmentare runtime pentru Exp 5
            if exp['aug']:
                data_augmentation = models.Sequential([
                    layers.GaussianNoise(0.1)
                ])
                train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

            # Optimizare cache
            train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            # Build Model
            model = build_model(lr=exp['lr'], dropout=exp['drop'], extra_layer=exp['extra'])
            
            # Train (5 epoci)
            start_time = time.time()
            history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
            end_time = time.time()
            
            duration = round(end_time - start_time, 2)
            
            # Extragere metrici (Acum numele sunt garantate 'val_precision', 'val_recall')
            val_acc = history.history['val_accuracy'][-1]
            try:
                val_prec = history.history['val_precision'][-1]
                val_recall = history.history['val_recall'][-1]
            except KeyError:
                # Fallback in caz ca TensorFlow decide sa puna '_1' totusi
                keys = list(history.history.keys())
                prec_key = [k for k in keys if 'precision' in k and 'val' in k][0]
                rec_key = [k for k in keys if 'recall' in k and 'val' in k][0]
                val_prec = history.history[prec_key][-1]
                val_recall = history.history[rec_key][-1]
            
            # Calcul F1
            if (val_prec + val_recall) > 0:
                f1 = 2 * (val_prec * val_recall) / (val_prec + val_recall)
            else:
                f1 = 0.0
            
            # Generare Observatii
            obs = "Referinta"
            if exp['id'] == "Exp 1": obs = "Convergenta mai lenta"
            elif exp['id'] == "Exp 2": obs = "Viteza mare, stabilitate redusa"
            elif exp['id'] == "Exp 3": obs = "Capacitate crescuta"
            elif exp['id'] == "Exp 4": obs = "Regularizare agresiva"
            elif exp['id'] == "Exp 5": obs = "Robustete (BEST)"

            print(f"   Rezultat: Acc={val_acc:.4f} | F1={f1:.4f} | Time={duration}s")
            
            # Scriere in CSV
            writer.writerow([exp['id'], exp['mod'], round(val_acc, 4), round(f1, 4), f"{duration}s", obs])

    print(f"\n Tabel experimente generat in: {csv_file}")

if __name__ == "__main__":
    main()