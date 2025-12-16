import os
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

# --- CONFIGURARE ---
app = Flask(__name__)

# Calea catre radacina proiectului
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# --- MODIFICARE CHEIE PENTRU ETAPA 5 ---
# Folosim modelul ANTRENAT (.h5), nu cel vechi (.keras)
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.h5")

# Incarcam modelul o singura data la pornire
print(f"[INFO] Incarcare model din: {MODEL_PATH}...")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model ANTRENAT incarcat cu succes!")
    except Exception as e:
        print(f"❌ EROARE la incarcarea modelului: {e}")
        model = None
else:
    print(f"❌ EROARE: Fisierul nu exista la {MODEL_PATH}")
    print("Ruleaza intai 'python src/neural_network/train.py' pentru a genera modelul!")
    model = None

def preprocess_image(img_path):
    """Pregateste imaginea pentru predictie (Grayscale -> Resize -> Normalize)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# --- INTERFATA HTML ---
HTML_PAGE = """
<!doctype html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <title>SLM AI Monitor - Etapa 5</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h2>🔍 SLM Defect Detection</h2>
        <p>Sistem de inspecție vizuală în timp real</p>
        <hr>
        <p>Încarcă imaginea stratului curent:</p>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file accept=".png, .jpg, .jpeg" required>
          <br>
          <input type=submit value="Start Inspecție">
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
        
        # Salvam temporar imaginea primita
        temp_path = os.path.join(BASE_DIR, "temp_upload.png")
        file.save(temp_path)
        
        if model:
            # Procesare si Predictie
            processed_img = preprocess_image(temp_path)
            if processed_img is None:
                return "Eroare: Imaginea nu a putut fi citita."
                
            score = model.predict(processed_img, verbose=0)[0][0]
            
            # Interpretare rezultat
            # Modelul a fost antrenat cu 'defect'=1 si 'ok'=0 (de obicei, sau invers in functie de folder order)
            # Verificam logica: Daca ai folosit image_dataset_from_directory, ordinea e alfabetica:
            # 0 = defect, 1 = ok.
            
            # ATENTIE: Ajusteaza logica daca rezultatele sunt inversate!
            # Presupunem standard: 0=Defect, 1=OK (sau invers). 
            # Daca binary_crossentropy + sigmoid -> output e probabilitatea clasei 1.
            
            # Vom afisa probabilitatea si clasa
            if score < 0.5:
                # Clasa 0
                result_text = "DEFECT DETECTAT (Clasa 0)"
                css_class = "defect"
                confidence = (1 - score) * 100
            else:
                # Clasa 1
                result_text = "PROCES STABIL / OK (Clasa 1)"
                css_class = "ok"
                confidence = score * 100
            
            # Stergem fisierul temporar
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return f"""
            <!doctype html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 50px; background-color: #f0f2f5; }}
                    .container {{ background: white; padding: 40px; border-radius: 15px; display: inline-block; max-width: 500px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                    .defect {{ color: #c62828; font-weight: bold; border: 2px solid #c62828; padding: 10px; border-radius: 5px; background: #ffebee; }}
                    .ok {{ color: #2e7d32; font-weight: bold; border: 2px solid #2e7d32; padding: 10px; border-radius: 5px; background: #e8f5e9; }}
                    a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Rezultat Inspecție</h2>
                    <br>
                    <h1 class="{css_class}">{result_text}</h1>
                    <p>Scor Model: <strong>{score:.4f}</strong></p>
                    <p>Încredere decizie: <strong>{confidence:.2f}%</strong></p>
                    <br>
                    <hr>
                    <br>
                    <a href='/'>⬅ Analizează alt strat</a>
                </div>
            </body>
            """
        else:
            return "Eroare: Modelul AI nu este încărcat corect."
            
    return HTML_PAGE

if __name__ == '__main__':
    print("🚀 Serverul pornește...")
    print("🌍 Deschide browserul la adresa: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)