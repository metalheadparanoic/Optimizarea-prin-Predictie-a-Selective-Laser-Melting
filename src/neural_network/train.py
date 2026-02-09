import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import sys

# Setup cai importuri
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.neural_network.model import build_cnn_model

# Configurare Cai
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Asiguram existenta folderelor
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30 # Punem mai multe, oricum EarlyStopping il opreste cand e gata

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Salvam graficul pentru documentatie
    plt.savefig(os.path.join(DOCS_DIR, "loss_curve.png"))
    # Salvam si in results (pentru Etapa 6)
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves_final.png"))
    print("Grafice salvate.")
    plt.close()

def main():
    print("Start Antrenare Model (Etapa 5/6)...")

    # Incarcare Date
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode='grayscale', label_mode='binary', shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode='grayscale', label_mode='binary'
    )

    # Optimizare viteza
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Construire Model (cel nou din model.py)
    model = build_cnn_model(input_shape=(64, 64, 1))

    # Callbacks
    callbacks = [
        # Opreste daca nu invata timp de 6 epoci
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        # Scade rata de invatare daca se blocheaza
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        # Salveaza istoricul intr-un CSV (Important pentru Results!)
        CSVLogger(os.path.join(RESULTS_DIR, "training_history.csv"))
    ]

    # Antrenare
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks
    )

    # Salvare Model
    # Salvam si ca 'trained' (pentru compatibilitate) si ca 'optimized' (pentru Etapa 6)
    model.save(os.path.join(MODELS_DIR, "trained_model.h5"))
    model.save(os.path.join(MODELS_DIR, "optimized_model.h5"))
    
    print("Model salvat cu succes.")
    plot_history(history)

if __name__ == "__main__":
    main()