import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys

# Setup cai
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test") # Folosim setul de TEST

# 1. Incarcam Modelul
model_path = os.path.join(MODELS_DIR, "trained_model.h5")
print(f"⌛ Incarcare model din {model_path}...")
model = tf.keras.models.load_model(model_path)

# 2. Pregatim datele de test
print("⌛ Incarcare date de test...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(64, 64),
    batch_size=32,
    color_mode='grayscale',
    label_mode='binary',
    shuffle=False 
)

# Extragem etichetele reale si predictiile
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    y_true.extend(labels.numpy().flatten())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds.flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)

# --- BONUS A: CONFUSION MATRIX ---
print("📸 Generare Matrice de Confuzie...")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Defect", "OK"])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SLM Model")
save_cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
plt.savefig(save_cm_path)
print(f"✅ Matrice salvata in: {save_cm_path}")

# --- BONUS B: TFLITE CONVERSION ---
print("📦 Conversie la TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(MODELS_DIR, "final_model.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ Model TFLite salvat in: {tflite_path}")

# --- BONUS C: BENCHMARK LATENTA ---
print("⏱️ Rulare Benchmark Latenta (100 iteratii)...")
# Pregatim o singura imagine dummy
input_shape = (1, 64, 64, 1)
dummy_input = tf.random.normal(input_shape)

# Incalzire
model.predict(dummy_input, verbose=0)

start_time = time.time()
for _ in range(100):
    model.predict(dummy_input, verbose=0)
end_time = time.time()

avg_time_ms = ((end_time - start_time) / 100) * 1000
print(f"🚀 Latenta medie: {avg_time_ms:.2f} ms")