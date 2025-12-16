import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_cnn_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

def plot_history(history):
    """
    Generează graficul de Loss/Accuracy și îl salvează în folderul DOCS.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    os.makedirs(DOCS_DIR, exist_ok=True)

    save_path = os.path.join(DOCS_DIR, "training_results.png")
    plt.savefig(save_path)
    print(f"Graficul a fost salvat în: {save_path}")

def main():
    print("Inițializare proces de antrenare...")

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Încărcare date din: {TRAIN_DIR}")
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )

    print("Construire model CNN")
    model = build_cnn_model(input_shape=(64, 64, 1))

    print("Start antrenare")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )

    os.makedirs(DOCS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "trained_model.h5")
    
    model.save(save_path)
    print(f"Model antrenat salvat cu succes in: {save_path}")

    plot_history(history)
    print("Proces finalizat cu succes.")

if __name__ == "__main__":
    main()