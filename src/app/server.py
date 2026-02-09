import os
from flask import Flask, request
import tensorflow as tf
import cv2
import numpy as np

# --- CONFIGURARE ---
app = Flask(__name__)

# Calea catre radacina proiectului
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Cautam modelul: Prioritate are cel Optimizat, apoi cel Antrenat standard
MODEL_PATH_OPT = os.path.join(BASE_DIR, "models", "optimized_model.h5")
MODEL_PATH_STD = os.path.join(BASE_DIR, "models", "trained_model.h5")

if os.path.exists(MODEL_PATH_OPT):
    LOAD_PATH = MODEL_PATH_OPT
    print(f" INFO: Se incarca modelul OPTIMIZAT din: {LOAD_PATH}")
elif os.path.exists(MODEL_PATH_STD):
    LOAD_PATH = MODEL_PATH_STD
    print(f" INFO: Se incarca modelul STANDARD din: {LOAD_PATH}")
else:
    LOAD_PATH = None
    print(" EROARE CRITICA: Nu s-a gasit niciun model .h5 in folderul models!")

# Incarcare model la pornire
model = None
if LOAD_PATH:
    try:
        model = tf.keras.models.load_model(LOAD_PATH)
        print(" Model incarcat cu succes. Serverul este gata.")
    except Exception as e:
        print(f" EROARE la incarcarea modelului: {e}")

def preprocess_image(img_path):
    """
    Pregateste imaginea pentru predictie.
    IMPORTANT: NU impartim la 255.0 aici, deoarece modelul are strat de Rescaling!
    """
    # Citire in Grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Redimensionare la 64x64 (dimensiunea cu care a fost antrenat)
    img = cv2.resize(img, (64, 64))
    
    # Adaugare dimensiuni pentru batch: (1, 64, 64, 1)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    return img

# --- INTERFATA HTML ---
HTML_PAGE = """
<!doctype html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <title>SLM AI Monitor - Etapa 6</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 50px; background-color: #f0f2f5; }
        .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; width: 100%; }
        h2 { color: #2c3e50; margin-bottom: 20px; }
        input[type=file] { margin: 20px 0; padding: 10px; border: 1px dashed #ccc; width: 80%; }
        input[type=submit] { background-color: #007bff; color: white; border: none; padding: 12px 25px; cursor: pointer; border-radius: 5px; font-size: 16px; transition: background 0.3s; }
        input[type=submit]:hover { background-color: #0056b3; }
        .result-box { margin-top: 20px; padding: 20px; border-radius: 8px; }
        .defect { background-color: #ffebee; color: #c62828; border: 2px solid #c62828; }
        .ok { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32; }
        .info { font-size: 0.9em; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>SLM Defect Detection</h2>
        <p>Incarca imaginea stratului metalic (Melt Pool)</p>
        <hr>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file accept=".png, .jpg, .jpeg" required>
          <br>
          <input type=submit value="Analizeaza Imagine">
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Niciun fisier selectat'
        file = request.files['file']
        if file.filename == '':
            return 'Niciun fisier selectat'
        
        # Salvare temporara
        temp_path = os.path.join(BASE_DIR, "temp_upload.png")
        file.save(temp_path)
        
        if model:
            # Procesare
            processed_img = preprocess_image(temp_path)
            
            if processed_img is None:
                return "Eroare: Imaginea nu a putut fi procesata."

            # Predictie
            # Outputul este intre 0 si 1 (Sigmoid)
            # Conform folderelor: 0 = defect, 1 = ok
            score = model.predict(processed_img, verbose=0)[0][0]
            
            # Interpretare
            if score < 0.5:
                # Este mai aproape de 0 -> DEFECT
                label = "DEFECT DETECTAT"
                css_class = "defect"
                confidence = (1 - score) * 100
            else:
                # Este mai aproape de 1 -> OK
                label = "PROCES STABIL / OK"
                css_class = "ok"
                confidence = score * 100
            
            # Curatenie
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Rezultat HTML
            return f"""
            <!doctype html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 50px; background-color: #f0f2f5; }}
                    .container {{ background: white; padding: 40px; border-radius: 15px; display: inline-block; max-width: 500px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                    .defect {{ color: #c62828; font-weight: bold; border: 2px solid #c62828; padding: 15px; border-radius: 8px; background: #ffebee; }}
                    .ok {{ color: #2e7d32; font-weight: bold; border: 2px solid #2e7d32; padding: 15px; border-radius: 8px; background: #e8f5e9; }}
                    a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Rezultat Analiza AI</h2>
                    <br>
                    <div class="{css_class}">
                        <h1>{label}</h1>
                        <p>Incredere Model: <strong>{confidence:.2f}%</strong></p>
                        <p style="font-size:0.8em; color:#555;">(Raw Score: {score:.4f})</p>
                    </div>
                    <br><br>
                    <a href='/'> Inapoi la incarcare</a>
                </div>
            </body>
            """
        else:
            return "Eroare: Modelul nu este incarcat."
            
    return HTML_PAGE

if __name__ == '__main__':
    print(" Server porneste pe http://127.0.0.1:5000")
    app.run(debug=True, port=5000)