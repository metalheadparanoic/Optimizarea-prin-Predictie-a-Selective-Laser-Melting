import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import sys

# Adaugam calea catre src pentru importuri
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Importam arhitectura ta
from src.neural_network.model import build_cnn_model

# --- CONFIGURARE CĂI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Asiguram existenta folderelor
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Hiperparametri
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50  # Punem mai multe, ca oricum il opreste EarlyStopping

def plot_history(history):
    """Genereaza si salveaza graficul de Loss/Accuracy (Cerinta Nivel 2)."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Loss (Curba de eroare)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # Salvare in docs/loss_curve.png
    save_path = os.path.join(DOCS_DIR, "loss_curve.png")
    plt.savefig(save_path)
    print(f"✅ Grafic salvat în: {save_path}")
    plt.close()

def main():
    print("🚀 Începere antrenare Nivel 2 (cu Callbacks)...")

    # 1. Incarcare date
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        label_mode='binary' # 0=Defect, 1=OK (sau invers, depinde de foldere)
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        label_mode='binary'
    )

    # Optimizare performanta (cache in memorie)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Construire model
    model = build_cnn_model(input_shape=(64, 64, 1))

    # 3. Definire Callbacks (NIVEL 2)
    callbacks_list = [
        # EarlyStopping: Opreste daca val_loss nu scade timp de 5 epoci
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        
        # ReduceLROnPlateau: Scade learning rate daca intram in platou
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # 4. Antrenare
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list # <--- Aici e magia Nivel 2
    )

    # 5. Salvare Model Final (.h5)
    save_path = os.path.join(MODELS_DIR, "trained_model.h5")
    model.save(save_path)
    print(f"✅ Model antrenat salvat cu succes în: {save_path}")

    # 6. Generare Grafic
    plot_history(history)
    
    print("🏁 Proces Nivel 2 finalizat.")

if __name__ == "__main__":
    main()