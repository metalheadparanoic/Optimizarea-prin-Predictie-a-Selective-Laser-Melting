import os
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

# --- CONFIGURARE ---
app = Flask(__name__)

# Calea catre radacina proiectului
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Calea catre modelul mutat in folderul 'models'
MODEL_PATH = os.path.join(BASE_DIR, "models", "slm_model.keras")

# Incarcam modelul o singura data la pornire
print(f"[INFO] Incarcare model din: {MODEL_PATH}...")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model incarcat cu succes!")
    except Exception as e:
        print(f"EROARE la incarcarea modelului: {e}")
        model = None
else:
    print(f"EROARE: Fisierul nu exista la {MODEL_PATH}")
    print("Verifica daca ai mutat fisierul .keras din 'docs' in 'models'!")
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
# O pagina simpla
HTML_PAGE = """
<!doctype html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <title>SLM AI Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f4f4f9; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }
        h2 { color: #333; }
        input[type=file] { margin: 20px 0; }
        input[type=submit] { background-color: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; }
        input[type=submit]:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .defect { background-color: #ffe6e6; color: #cc0000; border: 1px solid #cc0000; }
        .ok { background-color: #e6fffa; color: #006600; border: 1px solid #006600; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Sistem Inteligent de Monitorizare SLM</h2>
        <p>Încarcă o imagine a melt-pool-ului pentru analiză:</p>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file accept=".png, .jpg, .jpeg" required>
          <br>
          <input type=submit value="Analizează Imaginea">
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
            
            # Interpretare rezultat (0=Defect, 1=OK)
            if score < 0.5:
                result_text = "DEFECT DETECTAT"
                css_class = "defect"
                confidence = (1 - score) * 100
            else:
                result_text = "PROCES STABIL (OK)"
                css_class = "ok"
                confidence = score * 100
            
            # Stergem fisierul temporar
            os.remove(temp_path)

            return f"""
            <!doctype html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f4f4f9; }}
                    .container {{ background: white; padding: 30px; border-radius: 10px; display: inline-block; }}
                    .defect {{ color: red; }} .ok {{ color: green; }}
                    a {{ text-decoration: none; color: blue; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Rezultat Analiză</h2>
                    <h1 class="{css_class}">{result_text}</h1>
                    <p>Încredere model: <strong>{confidence:.2f}%</strong></p>
                    <p>Scor brut: {score:.4f}</p>
                    <br>
                    <a href='/'>⬅ Încarcă altă imagine</a>
                </div>
            </body>
            """
        else:
            return "Eroare: Modelul AI nu este încărcat."
            
    return HTML_PAGE

if __name__ == '__main__':
    print("Serverul pornește...")
    print("Deschide browserul la adresa: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)