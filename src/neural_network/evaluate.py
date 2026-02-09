import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import sys

# --- CONFIGURARE ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Asiguram folderele de iesire
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def main():
    print(" Incepere Evaluare Finala Completa...")

    # 1. Incarcare Model (Prioritate: Optimizat -> Antrenat)
    model_path = os.path.join(MODELS_DIR, "optimized_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "trained_model.h5")
    
    if not os.path.exists(model_path):
        print(" Eroare: Nu gasesc niciun model (.h5)!")
        return

    print(f" Incarcare model din: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2. Incarcare Date de Test
    # IMPORTANT: shuffle=False pentru Matricea de Confuzie
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(64, 64),
        batch_size=32,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=False
    )

    # 3. Evaluare Generala (Loss/Accuracy)
    print(" Calculare metrici globale...")
    results = model.evaluate(test_ds, verbose=1)
    test_loss, test_acc = results[0], results[1]
    print(f" Rezultate Globale: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")

    # Salvare JSON
    metrics_data = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "model_name": "SLM_CNN_Optimized",
        "status": "Production Ready"
    }
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # 4. Generare MATRICE DE CONFUZIE (Partea Vizuala)
    print(" Generare Matrice de Confuzie...")
    
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        y_true.extend(labels.numpy().flatten())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
    
    # Numele claselor (in functie de ordinea alfabetica a folderelor)
    class_names = test_ds.class_names # ['defect', 'ok']
    
    # Desenare Matrice
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix (Acc: {test_acc:.2%})")
    
    # Salvare PNG in DOCS
    save_cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
    plt.savefig(save_cm_path)
    print(f" Matrice salvata in: {save_cm_path}")

    # 5. Raport Detaliat (Precision/Recall)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + "="*40)
    print(" RAPORT DETALIAT PE CLASE")
    print("="*40)
    print(report)

    # Salvare raport text
    with open(os.path.join(RESULTS_DIR, "final_classification_report.txt"), "w") as f:
        f.write(report)

    # 6. Generare Hyperparameters YAML 
    yaml_content = """
model_type: "CNN Custom Architecture"
input_shape: [64, 64, 1]
preprocessing: "Industrial Noise + Vignette + Shift"
optimizer: "Adam"
callbacks:
  - "EarlyStopping"
  - "ReduceLROnPlateau"
  - "CSVLogger"
    """
    with open(os.path.join(RESULTS_DIR, "hyperparameters.yaml"), 'w') as f:
        f.write(yaml_content.strip())

    print(" Evaluare completa! Verifica folderul 'docs' si 'results'.")

if __name__ == "__main__":
    main()