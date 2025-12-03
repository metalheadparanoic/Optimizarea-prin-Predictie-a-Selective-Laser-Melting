import os
import cv2
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
IMG_SIZE = (64, 64)

def create_dirs():
    for cls in ["ok", "defect"]:
        os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)

def process_and_save():
    classes = ["ok", "defect"]
    count = 0
    
    print(f"Procesare imagini din {RAW_DIR} catre {PROCESSED_DIR}...")
    
    for cls in classes:
        src_path = os.path.join(RAW_DIR, cls)
        dst_path = os.path.join(PROCESSED_DIR, cls)

        for filename in os.listdir(src_path):
            if filename.endswith(".png"):

                img_path = os.path.join(src_path, filename)
                img = cv2.imread(img_path)

                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, IMG_SIZE)

                cv2.imwrite(os.path.join(dst_path, filename), img_resized)
                count += 1
                
    print(f"Finalizat. {count} imagini procesate si salvate in 'data/processed'.")

if __name__ == "__main__":
    create_dirs()
    process_and_save()