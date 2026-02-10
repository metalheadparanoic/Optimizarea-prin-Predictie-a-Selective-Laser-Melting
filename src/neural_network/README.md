# Modul 2: Rețea Neuronală Convoluțională (CNN)

Acest modul conține logica centrală de Inteligență Artificială a proiectului. Aici sunt definite arhitectura rețelei neuronale, procesele de antrenare, optimizare a hiperparametrilor și evaluare a performanței.

Sistemul utilizează **TensorFlow** și **Keras** pentru a construi un clasificator binar de imagini capabil să distingă între un melt pool stabil (OK) și unul instabil (Defect).

## Arhitectura Modelului (`model.py`)

Modelul este o Rețea Neuronală Convoluțională (CNN) secvențială, proiectată pentru eficiență pe imagini de dimensiuni reduse (64x64 pixeli).

Componente principale:
1.  **Input:** Imagini Grayscale (64x64x1).
2.  **Preprocesare Integrată:** Straturi de `Rescaling` (normalizare [0,1]) și Augmentare (Flip, Rotation, Zoom) incluse direct în model.
3.  **Extragere Trăsături:** 3 blocuri de convoluție (`Conv2D` + `MaxPooling2D`) cu număr crescător de filtre (32 -> 64 -> 128).
4.  **Clasificare:** Strat `Flatten`, urmat de un strat dens (`Dense` 128 neuroni), `Dropout` (0.5) pentru regularizare și stratul final `Dense` (1 neuron) cu activare `Sigmoid`.

## Descrierea Scripturilor

### `train.py`
Scriptul standard de antrenare.
* **Funcție:** Antrenează modelul o singură dată cu o configurație de bază.
* **Output:** Salvează modelul antrenat în `models/trained_model.h5` și istoricul antrenării în `results/training_history.csv`.

### `optimize.py`
Scriptul de optimizare automată (Grid Search).
* **Funcție:** Rulează multiple experimente variind hiperparametrii (Learning Rate, Batch Size, Arhitectură, Augmentare).
* **Scop:** Identifică configurația care maximizează acuratețea și robustețea.
* **Output:** Salvează cel mai performant model în `models/optimized_model.h5` și un raport comparativ în `results/optimization_experiments.csv`.

### `evaluate.py`
Scriptul de validare finală.
* **Funcție:** Încarcă modelul salvat și îl testează pe setul de date de Test (care nu a fost văzut la antrenare).
* **Output:** Generează metrici finale (Accuracy, Precision, Recall, F1-Score) și le salvează în `results/final_metrics.json`.

### `predict.py`
Modulul de inferență.
* **Funcție:** Permite clasificarea unor imagini individuale noi.
* **Utilizare:** Este apelat de interfața grafică sau din linia de comandă pentru a verifica piese specifice.

### `visualize.py`
Generatorul de rapoarte grafice.
* **Funcție:** Preia log-urile de antrenare și generează vizualizări pentru interpretarea rezultatelor.
* **Grafice generate:**
    * Curbe de învățare (Loss/Accuracy) -> `docs/results/learning_curves.png`
    * Matricea de confuzie -> `docs/results/confusion_matrix.png`

### `save_untrained.py`
Utilitar tehnic.
* **Funcție:** Salvează o versiune a modelului doar cu greutățile inițializate random. Folosit pentru a demonstra diferența de performanță dintre un model "gol" și unul antrenat.

## Fluxul de Lucru Recomandat

1.  **Antrenare Baseline:** Rulați `train.py` pentru a verifica funcționalitatea datelor.
2.  **Optimizare:** Rulați `optimize.py` (poate dura câteva minute) pentru a găsi cel mai bun model.
3.  **Vizualizare:** Rulați `visualize.py` pentru a genera graficele de performanță.
4.  **Evaluare Finală:** Rulați `evaluate.py` pentru a confirma metricile pe setul de test.