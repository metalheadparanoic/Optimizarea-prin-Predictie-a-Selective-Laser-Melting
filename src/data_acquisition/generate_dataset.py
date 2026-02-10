import os
import cv2
import numpy as np
import random

# --- CONFIGURARE ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
IMG_SIZE = 64
SAMPLES_PER_CLASS = 1000  # Generam suficiente date pentru a obtine acuratete mare

# Creare foldere
os.makedirs(os.path.join(RAW_DATA_DIR, "ok"), exist_ok=True)
os.makedirs(os.path.join(RAW_DATA_DIR, "defect"), exist_ok=True)

def add_noise(image):
    """Adauga zgomot Gaussian pentru a simula senzorul camerei."""
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss * 50
    return np.clip(noisy, 0, 255).astype(np.uint8)

def generate_stable_meltpool(index):
    """Genereaza o imagine 'OK' - elipsa regulata, centrata."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    # Parametri fizici simulati (stabilitate)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    axes = (random.randint(10, 14), random.randint(8, 12)) # Variatie mica
    angle = random.randint(0, 180)
    
    # Desenare elipsa alba (Melt Pool)
    cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)
    
    # Adaugare gradient termic (simulat prin blur)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = add_noise(img)
    
    # Salvare
    path = os.path.join(RAW_DATA_DIR, "ok", f"ok_{index}.png")
    cv2.imwrite(path, img)

def generate_unstable_meltpool(index):
    """Genereaza o imagine 'DEFECT' - forma neregulata, spatter, intreruperi."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    
    # 1. Forma de baza distorsionata
    axes = (random.randint(8, 16), random.randint(6, 14))
    angle = random.randint(0, 360)
    cv2.ellipse(img, center, axes, angle, 0, 360, 200, -1)
    
    # 2. Simulare 'Keyhole collapse' sau instabilitate (pete negre interne)
    if random.random() > 0.5:
        offset_x = random.randint(-5, 5)
        offset_y = random.randint(-5, 5)
        cv2.circle(img, (center[0]+offset_x, center[1]+offset_y), random.randint(2, 4), 0, -1)

    # 3. Simulare 'Spatter' (stropi metalici in jur)
    for _ in range(random.randint(3, 8)):
        spat_x = random.randint(10, IMG_SIZE-10)
        spat_y = random.randint(10, IMG_SIZE-10)
        # Ne asiguram ca nu e chiar in centru
        if abs(spat_x - center[0]) > 10 or abs(spat_y - center[1]) > 10:
            cv2.circle(img, (spat_x, spat_y), 1, 255, -1)

    # 4. Deformare contur (simulare suprafata rugoasa)
    # Desenam un poligon aleator peste elipsa
    pts = np.array([[random.randint(20, 44), random.randint(20, 44)] for _ in range(5)], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, 0, 1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = add_noise(img)
    
    # Salvare
    path = os.path.join(RAW_DATA_DIR, "defect", f"defect_{index}.png")
    cv2.imwrite(path, img)

def main():
    print(f"[INFO] Generare dataset sintetic in: {RAW_DATA_DIR}")
    print(f"[INFO] Target: {SAMPLES_PER_CLASS} imagini per clasa.")
    
    for i in range(SAMPLES_PER_CLASS):
        generate_stable_meltpool(i)
        generate_unstable_meltpool(i)
        
        if (i+1) % 500 == 0:
            print(f" -> Generat {i+1} / {SAMPLES_PER_CLASS} perechi...")
            
    print("[OK] Generare completa.")

if __name__ == "__main__":
    main()