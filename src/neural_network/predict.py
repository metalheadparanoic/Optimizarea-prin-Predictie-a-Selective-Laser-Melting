import os
import cv2
import numpy as np
import tensorflow as tf
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
MODEL_PATH = os.path.join(PROJECT_ROOT, "docs", "slm_model.keras")

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Nu gasesc modelul la: {MODEL_PATH}")
        return

    print(f"Incarcare model din: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    true_label = random.choice(["ok", "defect"]) 
    folder_path = os.path.join(TEST_DIR, true_label)
    filename = random.choice(os.listdir(folder_path))
    image_path = os.path.join(folder_path, filename)

    print(f"\n--- TESTARE PE IMAGINE NOUA ---")
    print(f"Imagine selectata: {filename}")
    print(f"Eticheta reala: {true_label.upper()}")

    processed_img = load_and_preprocess_image(image_path)
    prediction_score = model.predict(processed_img, verbose=0)[0][0]
  
    if prediction_score < 0.5:
        predicted_label = "DEFECT"
        confidence = (1 - prediction_score) * 100
    else:
        predicted_label = "OK"
        confidence = prediction_score * 100

    print(f"Modelul zice: {predicted_label}")
    print(f"Incredere: {confidence:.2f}%")
    print(f"Scor brut: {prediction_score:.4f}")

    if predicted_label.lower() == true_label:
        print("\nREZULTAT CORECT")
    else:
        print("\nGRESIT")

if __name__ == "__main__":
    main()