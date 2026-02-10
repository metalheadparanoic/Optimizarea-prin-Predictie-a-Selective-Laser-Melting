import os
import argparse
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- REPARARE IMPORTURI ---
# Adaugam folderul radacina (PROIECT) in sys.path
# Astfel Python stie ca "src" este un pachet valid
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/neural_network
project_root = os.path.abspath(os.path.join(current_dir, "../../")) # PROIECT
sys.path.append(project_root)

# Acum importam direct (fara try-except, ca sa vedem eroarea reala daca exista)
from src.neural_network.model import create_model

# --- CONFIGURARE CAI ---
DATA_DIR = os.path.join(project_root, "data")
MODELS_DIR = os.path.join(project_root, "models")
RESULTS_DIR = os.path.join(project_root, "results")

# Creare foldere necesare
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model(lr, batch_size, epochs, dropout_rate, exp_name):
    print(f"\n[INFO] Start Antrenament: {exp_name}")
    print(f"       Parametri: LR={lr}, BS={batch_size}, Epochs={epochs}, Dropout={dropout_rate}")

    # 1. Incarcare Date
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"[EROARE] Folderele de date nu exista: {train_dir}")
        print("       Rulati intai generate_dataset.py si processed_data.py!")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(64, 64),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(64, 64),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=False
    )

    # Optimizare performanta dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Creare Model
    # Acum apelam functia importata din model.py
    model = create_model(dropout_rate)
    
    # Compilare cu Learning Rate specificat
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 3. Callbacks
    model_name = f"trained_model_{exp_name}.h5" if exp_name != "default" else "trained_model.h5"
    checkpoint_path = os.path.join(MODELS_DIR, model_name)
    
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # 4. Antrenare
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )

    print(f"[OK] Antrenament finalizat. Model salvat: {checkpoint_path}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script antrenare SLM Neural Network")
    
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--batch", type=int, default=32, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=25, help="Numar epoci")
    parser.add_argument("--dropout", type=float, default=0.5, help="Rata Dropout")
    parser.add_argument("--name", type=str, default="default", help="Nume experiment (sufix fisier)")

    args = parser.parse_args()

    train_model(
        lr=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        dropout_rate=args.dropout,
        exp_name=args.name
    )