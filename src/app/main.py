import os
import sys
import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# --- CONFIGURARE CAI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "temp_uploads")

# Asigurare foldere
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- INCARCARE MODEL ---
model = None
model_name = "N/A"

def load_best_model():
    global model, model_name
    
    # Ordine prioritate: Optimizat -> Antrenat (Standard)
    opt_path = os.path.join(MODELS_DIR, "optimized_model.h5")
    std_path = os.path.join(MODELS_DIR, "trained_model.h5")
    
    path_to_load = None
    
    if os.path.exists(opt_path):
        path_to_load = opt_path
        model_name = "Optimized Model (Etapa 6)"
        print(f"[INFO] Incarcare model OPTIMIZAT: {opt_path}")
    elif os.path.exists(std_path):
        path_to_load = std_path
        model_name = "Standard Model (Etapa 5)"
        print(f"[WARN] Incarcare model STANDARD (Baseline): {std_path}")
    else:
        print("[EROARE] Nu am gasit niciun model .h5!")
        return False

    try:
        model = tf.keras.models.load_model(path_to_load)
        print("[OK] Model incarcat cu succes.")
        return True
    except Exception as e:
        print(f"[EROARE] {e}")
        return False

# Incarcam modelul la start
load_best_model()

# --- PREPROCESARE ---
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    return img_array

# --- LOGGING ---
def log_prediction(filename, raw_score, label):
    log_path = os.path.join(RESULTS_DIR, "production_log.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Scriere header daca nu exista
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,filename,raw_score,label,model_version\n")
            
    with open(log_path, "a") as f:
        f.write(f"{timestamp},{filename},{raw_score:.6f},{label},{model_name}\n")

# --- RUTE FLASK ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    confidence_text = ""
    color_class = ""
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', msg='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', msg='No selected file')
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predictie
            if model:
                img_array = preprocess_image(filepath)
                prediction = model.predict(img_array)
                score = prediction[0][0] # Probabilitate clasa 1 (OK)
                
                # Interpretare (Threshold 0.5)
                # 0 = Defect, 1 = OK
                if score > 0.5:
                    label = "OK"
                    color_class = "success" # CSS class pentru verde
                    confidence = score * 100
                else:
                    label = "DEFECT"
                    color_class = "danger"  # CSS class pentru rosu
                    confidence = (1 - score) * 100
                
                prediction_text = f"REZULTAT: {label}"
                confidence_text = f"Încredere: {confidence:.2f}% (Raw: {score:.4f})"
                
                # Logare
                log_prediction(filename, score, label)
            else:
                prediction_text = "Eroare: Modelul nu este incarcat."

    # HTML simplu integrat (sau poti folosi templates/index.html separat)
    html_template = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>SLM Defect Detection</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      </head>
      <body class="bg-light">
        <div class="container py-5">
            <div class="card shadow">
                <div class="card-header bg-primary text-white">
                    <h3 class="mb-0">🔍 SLM Quality Control AI</h3>
                    <small>{model_name}</small>
                </div>
                <div class="card-body text-center">
                    <form method="post" enctype="multipart/form-data" class="mb-4">
                        <div class="mb-3">
                            <input class="form-control" type="file" name="file" required>
                        </div>
                        <button type="submit" class="btn btn-primary btn-lg">Analizează Imaginea</button>
                    </form>
                    
                    {'<div class="alert alert-' + color_class + '">' if prediction_text else ''}
                        <h2 class="display-4">{prediction_text}</h2>
                        <p class="lead">{confidence_text}</p>
                    {'</div>' if prediction_text else ''}
                </div>
                <div class="card-footer text-muted">
                    Sistem activ. Logs salvate în results/production_log.csv
                </div>
            </div>
        </div>
      </body>
    </html>
    """
    
    return html_template

if __name__ == '__main__':
    # Rulare server
    print("\n[INFO] Pornire Server Flask...")
    print(f"[INFO] Accesati in browser: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)