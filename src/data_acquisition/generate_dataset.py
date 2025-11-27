import cv2
import numpy as np
import os
import random
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURARE CĂI (Relative la locul unde rulăm scriptul) ---
# Scriptul e în src/data_acquisition, deci urcăm 2 nivele (../../) până la PROIECT
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = 64
SAMPLES_PER_CLASS = 1000  # 1000 OK + 1000 Defecte = 2000 total

def create_dirs():
    """Creează structura de foldere dacă nu există."""
    print(f"Directoarele vor fi create în: {BASE_DIR}")
    for d in [os.path.join(RAW_DIR, "ok"), os.path.join(RAW_DIR, "defect")]:
        os.makedirs(d, exist_ok=True)
    
    classes = ["ok", "defect"]
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

def generate_melt_pool(is_defect=False):
    """Desenează un melt-pool simulat."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    
    if not is_defect:
        # --- OK ---
        radius = random.randint(10, 14)
        axes = (radius, radius + random.randint(-1, 1))
        angle = random.randint(0, 360)
        cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)
        img = cv2.GaussianBlur(img, (9, 9), 0)
    else:
        # --- DEFECT ---
        axis_x = random.randint(8, 18)
        axis_y = random.randint(5, 12)
        angle = random.randint(0, 180)
        cv2.ellipse(img, center, (axis_x, axis_y), angle, 0, 360, 255, -1)
        
        # Stropi (Spatter)
        num_spatter = random.randint(3, 8)
        for _ in range(num_spatter):
            sx = random.randint(10, 54)
            sy = random.randint(10, 54)
            if np.linalg.norm(np.array([sx, sy]) - np.array(center)) > 10:
                cv2.circle(img, (sx, sy), random.randint(1, 2), 200, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Zgomot
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img

def split_and_copy():
    """Împarte datele raw în train/val/test."""
    classes = ["ok", "defect"]
    for cls in classes:
        src_path = os.path.join(RAW_DIR, cls)
        # Listăm doar fișierele .png
        images = [f for f in os.listdir(src_path) if f.endswith('.png')]
        random.shuffle(images)
        
        # Împărțire: 70% Train, 15% Val, 15% Test
        train_imgs, test_val_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(test_val_imgs, test_size=0.5, random_state=42)
        
        print(f"Clasa '{cls}': {len(train_imgs)} Train, {len(val_imgs)} Val, {len(test_imgs)} Test")
        
        # Copiere
        for f in train_imgs: shutil.copy2(os.path.join(src_path, f), os.path.join(TRAIN_DIR, cls, f))
        for f in val_imgs: shutil.copy2(os.path.join(src_path, f), os.path.join(VAL_DIR, cls, f))
        for f in test_imgs: shutil.copy2(os.path.join(src_path, f), os.path.join(TEST_DIR, cls, f))

if __name__ == "__main__":
    print("Start Generare Date SLM...")
    create_dirs()
    
    # 1. Generare Raw
    print("Generare imagini brute (RAW)...")
    for i in range(SAMPLES_PER_CLASS):
        cv2.imwrite(os.path.join(RAW_DIR, "ok", f"ok_{i:04d}.png"), generate_melt_pool(False))
        cv2.imwrite(os.path.join(RAW_DIR, "defect", f"defect_{i:04d}.png"), generate_melt_pool(True))
        
    # 2. Împărțire
    print("scissors  Împărțire dataset (Train/Val/Test)...")
    split_and_copy()
    
    print("\n✅ GATA! Datele au fost generate local în folderul 'data'.")