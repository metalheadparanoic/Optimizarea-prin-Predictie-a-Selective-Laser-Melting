import os
import numpy as np
# import cv2 (va fi folosit acasa)

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    Incarca o imagine, o redimensioneaza si o normalizeaza.
    
    Args:
        image_path (str): Calea catre imagine.
        target_size (tuple): Dimensiunea tinta (width, height).
        
    Returns:
        np.array: Imaginea normalizata (valori 0-1).
    """
    # TODO: Implementare cu OpenCV
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, target_size)
    # return img / 255.0
    pass

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Imparte setul de date in folderele train/val/test.
    """
    # TODO: Implementare logica de mutare a fisierelor
    pass