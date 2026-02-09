import os
import argparse
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- CONFIGURARE CAI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Asiguram existenta folderelor
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def evaluate_model(model_path, detailed=False):
    print(f"\n[INFO] Start Evaluare pentru: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[EROARE] Fisierul modelului nu exista: {model_path}")
        sys.exit(1)

    # 1. Incarcare Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"[EROARE] Nu s-a putut incarca modelul: {e}")
        return

    # 2. Incarcare Date Test
    print("[INFO] Incarcare set de date Test...")
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            image_size=(64, 64),
            batch_size=32,
            color_mode='grayscale',
            label_mode='binary',
            shuffle=False 
        )
    except Exception as e:
        print(f"[EROARE] Nu s-a putut citi datasetul: {e}")
        return

    # 3. Predictii
    print("[INFO] Generare predictii...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        y_true.extend(labels.numpy().flatten())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    class_names = test_ds.class_names

    # 4. Matrice de Confuzie
    cm_filename = "confusion_matrix_optimized.png" if "optimized" in model_path else "confusion_matrix.png"
    print(f"[INFO] Salvare matrice confuzie: {cm_filename}")
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix\n({os.path.basename(model_path)})")
    plt.savefig(os.path.join(DOCS_DIR, cm_filename))
    plt.close()

    # 5. Raport Clasificare
    print("[INFO] Generare raport metrics...")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    with open(os.path.join(RESULTS_DIR, "final_classification_report.txt"), "w") as f:
        f.write(report)

    # 6. Analiza Erorilor (Daca e cerut detailed)
    if detailed:
        print("[INFO] Generare analiza erori (Top Errors)...")
        # Obtinem caile fisierelor (doar daca generatorul suporta file_paths)
        file_paths = test_ds.file_paths
        errors = []
        
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                error_info = {
                    "index": i,
                    "filename": os.path.basename(file_paths[i]),
                    "true_label": class_names[y_true[i]],
                    "predicted_label": class_names[y_pred[i]],
                    "confidence": float(y_pred_probs[i]),
                    "raw_score": float(y_pred_probs[i])
                }
                errors.append(error_info)
        
        error_path = os.path.join(RESULTS_DIR, "error_analysis.json")
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=4)
        print(f"[OK] Analiza erorilor salvata in: {error_path}")

    # 7. Conversie TFLite (doar daca e modelul optimizat)
    if "optimized" in model_path:
        print("\n[INFO] Conversie TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(os.path.join(MODELS_DIR, "final_model.tflite"), "wb") as f:
            f.write(tflite_model)
        print("[OK] Model TFLite exportat.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Argument optional: daca nu e dat, cauta automat optimized -> trained
    parser.add_argument("--model", type=str, help="Calea catre fisierul .h5")
    parser.add_argument("--detailed", action="store_true", help="Genereaza JSON cu erori")
    
    args = parser.parse_args()
    
    target_path = args.model
    
    # Logica automata daca nu se da path
    if target_path is None:
        opt = os.path.join(MODELS_DIR, "optimized_model.h5")
        std = os.path.join(MODELS_DIR, "trained_model.h5")
        if os.path.exists(opt):
            target_path = opt
        elif os.path.exists(std):
            target_path = std
        else:
            print("[EROARE] Niciun model gasit automat. Specificati --model path/to/model.h5")
            sys.exit(1)
            
    evaluate_model(target_path, detailed=args.detailed)