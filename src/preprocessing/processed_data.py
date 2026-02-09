import os
import cv2
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURARE CĂI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

def apply_industrial_camera_effect(img):
    """
    Simuleaza imperfectiunile camerei:
    1. Deplasare aleatoare (Shift) - ca piesa sa nu fie perfect centrata.
    2. Zgomot de senzor (Gaussian Noise).
    3. Vinietare (Colturi intunecate).
    """
    rows, cols = img.shape
    
    # 1. Random Shift (+/- 5 pixeli)
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # Umplem marginea ramasa goala cu culoarea medie a imaginii (gri)
    img = cv2.warpAffine(img, M, (cols, rows), borderValue=int(np.mean(img)))

    # 2. Zgomot (Fara pureci albi/negri)
    img_float = img.astype(np.float32)
    noise = np.random.normal(0, 18, img.shape).astype(np.float32)
    img_noisy = cv2.add(img_float, noise)
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    # 3. Vinietare
    kernel_x = cv2.getGaussianKernel(cols, int(cols * 0.8)) 
    kernel_y = cv2.getGaussianKernel(rows, int(rows * 0.8))
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    return (img_noisy * mask).astype(np.uint8)

def create_dirs(path):
    """Sterge si recreeaza structura de foldere."""
    if os.path.exists(path):
        try: shutil.rmtree(path)
        except Exception as e: print(f"⚠️ Atentie: {e}")
    os.makedirs(path, exist_ok=True)
    for cls in ["ok", "defect"]:
        os.makedirs(os.path.join(path, cls), exist_ok=True)

def process_and_split():
    print(" START: Procesare & Impartire Date...")
    print("-" * 50)
    
    # --- PASUL 1: GENERARE PROCESSED ---
    create_dirs(PROCESSED_DIR)
    
    classes = ["ok", "defect"]
    total_processed = 0

    print(" Aplicare efecte industriale...")
    for cls in classes:
        src_path = os.path.join(RAW_DIR, cls)
        dst_path = os.path.join(PROCESSED_DIR, cls)
        
        count_cls = 0
        if not os.path.exists(src_path):
            print(f" EROARE: Nu gasesc folderul {src_path}")
            continue

        for filename in os.listdir(src_path):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(src_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                img_processed = apply_industrial_camera_effect(img)
                cv2.imwrite(os.path.join(dst_path, filename), img_processed)
                count_cls += 1
        
        print(f"    Clasa '{cls.upper()}': {count_cls} imagini procesate.")
        total_processed += count_cls

    # --- PASUL 2: IMPARTIRE TRAIN/VAL/TEST ---
    print("-" * 50)
    print(" Impartire Dataset (Train / Val / Test)...")
    
    create_dirs(TRAIN_DIR)
    create_dirs(VAL_DIR)
    create_dirs(TEST_DIR)

    for cls in classes:
        src_path = os.path.join(PROCESSED_DIR, cls)
        all_images = [f for f in os.listdir(src_path) if f.endswith(".png")]
        
        # Amestecam imaginile inainte de split
        random.shuffle(all_images)
        
        # Split: 70% Train, 15% Val, 15% Test
        train_imgs, test_val = train_test_split(all_images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(test_val, test_size=0.5, random_state=42)
        
        # Copiere efectiva
        for f in train_imgs: shutil.copy2(os.path.join(src_path, f), os.path.join(TRAIN_DIR, cls, f))
        for f in val_imgs:   shutil.copy2(os.path.join(src_path, f), os.path.join(VAL_DIR, cls, f))
        for f in test_imgs:  shutil.copy2(os.path.join(src_path, f), os.path.join(TEST_DIR, cls, f))
        
        # --- AFISARE STATISTICI ---
        print(f"\n Statistici Clasa {cls.upper()}:")
        print(f"    Train: {len(train_imgs):<5} (Invatare)")
        print(f"    Val:   {len(val_imgs):<5} (Verificare in timpul invatarii)")
        print(f"    Test:  {len(test_imgs):<5} (Evaluare finala)")
        print(f"    Total: {len(all_images)}")

    print("-" * 50)
    print(" Procesare completa! Datele sunt gata de antrenare.")

if __name__ == "__main__":
    process_and_split()