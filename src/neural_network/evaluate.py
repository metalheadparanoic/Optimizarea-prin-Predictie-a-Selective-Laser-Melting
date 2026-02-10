import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- CONFIGURARE CAI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Asigurare foldere
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def evaluate_model(model_path, detailed=False):
    print(f"\n[INFO] Start Evaluare pentru: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"[EROARE] Modelul nu exista: {model_path}")
        return

    # 1. Incarcare Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"[EROARE] Nu pot incarca modelul: {e}")
        return

    # 2. Incarcare Date Test
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_dir):
        print(f"[EROARE] Folderul de test nu exista: {test_dir}")
        return

    print("[INFO] Incarcare set de date Test...")
    
    # Folosim image_dataset_from_directory pentru a incarca imaginile
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(64, 64),
        batch_size=32,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=False # IMPORTANT: Nu amestecam pentru a pastra ordinea etichetelor
    )

    # Extragem etichetele reale si predictiile
    y_true = []
    y_pred_probs = []
    
    print("[INFO] Generare predictii...")
    for images, labels in test_ds:
        # Predictie batch
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())

    # Conversie la numpy array
    y_true = np.array(y_true).flatten().astype(int) # Fortam intregi
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int) # Threshold 0.5

    class_names = test_ds.class_names # ['defect', 'ok']

    # 3. Matrice de Confuzie
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confuzie')
    plt.ylabel('Real')
    plt.xlabel('Predictie')
    
    # Salvam in docs/
    cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Salvare matrice confuzie: {cm_path}")

    # 4. Raport Clasificare
    print("[INFO] Generare raport metrics...")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Salvare raport in results/
    with open(os.path.join(RESULTS_DIR, "final_classification_report.txt"), "w") as f:
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(report)

    # 5. Analiza Erori (Top Errors) - Doar daca detailed=True
    if detailed:
        print("[INFO] Generare analiza erori (Top Errors)...")
        errors = []
        
        # Cautam indecsii unde predictia a fost gresita
        wrong_indices = np.where(y_true != y_pred)[0]
        
        # Daca nu sunt erori (cazul tau fericit!)
        if len(wrong_indices) == 0:
            print("[INFO] FELICITARI! Modelul nu a facut nicio greseala pe setul de test.")
            error_data = {"status": "perfect_score", "errors": []}
        else:
            # Calculam "cat de mult" a gresit (distanta fata de threshold)
            diffs = np.abs(y_pred_probs[wrong_indices] - y_true[wrong_indices])
            # Sortam descrescator dupa marimea erorii
            top_wrong = wrong_indices[np.argsort(diffs)[::-1]][:5] 

            for i in top_wrong:
                errors.append({
                    "index": int(i),
                    "true_label": class_names[int(y_true[i])], # <--- AICI ERA EROAREA (acum e int)
                    "pred_prob": float(y_pred_probs[i]),
                    "pred_label": class_names[int(y_pred[i])]
                })
            
            error_data = {"errors": errors}

        # Salvare JSON
        with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w") as f:
            json.dump(error_data, f, indent=4)

    # 6. Conversie TFLite (Bonus pentru productie)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(MODELS_DIR, "final_model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[INFO] Model convertit TFLite: {tflite_path}")
    except Exception as e:
        print(f"[WARN] Conversia TFLite a esuat: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Cale catre model specific")
    parser.add_argument("--detailed", action="store_true", help="Genereaza analiza erorilor")
    
    args = parser.parse_args()
    
    # Logica de selectie model
    target_path = ""
    if args.model:
        target_path = args.model
    else:
        # Cautam automat cel mai bun model
        opt_path = os.path.join(MODELS_DIR, "optimized_model.h5")
        std_path = os.path.join(MODELS_DIR, "trained_model.h5")
        
        if os.path.exists(opt_path):
            target_path = opt_path
        elif os.path.exists(std_path):
            target_path = std_path
        else:
            print("[EROARE] Niciun model gasit automat. Specificati --model path/to/model.h5")
            sys.exit(1)

    evaluate_model(target_path, detailed=args.detailed)