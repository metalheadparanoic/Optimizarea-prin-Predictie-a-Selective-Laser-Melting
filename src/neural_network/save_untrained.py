import os
import tensorflow as tf
import sys

# Ne asiguram ca putem importa din acelasi folder
sys.path.append(os.path.dirname(__file__))

# IMPORTUL CORECT: Folosim numele functiei tale
from model import build_cnn_model 

# --- CONFIGURARE CĂI ---
# Mergem 2 nivele mai sus pentru a ajunge la radacina PROIECT
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ne asiguram ca folderul models exista
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("Generare model NEANTRENAT (schelet arhitectură)...")
    
    # 1. Construim arhitectura (ponderile sunt initializate random)
    try:
        model = build_cnn_model()
        print("Arhitectura a fost construită cu succes.")
    except Exception as e:
        print(f"Eroare la construirea modelului: {e}")
        return

    # 2. Il salvam direct, fara antrenare (.fit)
    save_path = os.path.join(MODELS_DIR, "untrained_model.h5")
    model.save(save_path)
    
    print(f"Modelul neantrenat a fost salvat în: {save_path}")
    
if __name__ == "__main__":
    main()