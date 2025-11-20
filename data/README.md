## 1. Descrierea Setului de Date

### 1.1 Sursa datelor
* **Origine:** Date sintetice generate local.
* **Modul de achiziție:** Generare programatică (Script Python cu OpenCV).
* **Justificare:** Deoarece accesul la o mașină SLM reală cu monitorizare in-situ este limitat, vom simula defectele prin generare de poze.

### 1.2 Caracteristicile dataset-ului
* **Număr estimat de observații:** 100 imagini.
* **Număr de caracteristici (features):** 4096 (pentru o imagine 64x64 pixeli).
* **Tipuri de date:** Imagini (Matrice de pixeli).
* **Format fișiere:** PNG.

### 1.3 Descrierea caracteristicilor
| Caracteristică | Tip | Unitate | Descriere | Domeniu valori |
|---|---|---|---|---|
| Pixel Intensity | Numeric | - | Intensitatea luminoasă a melt-pool-ului | 0–255 (Grayscale) |
| Label (Target) | Categorial | - | Clasificarea stării (OK / Defect) | {0, 1} |

---

## 2. Analiza Exploratorie a Datelor (EDA) – Planificare

*Deoarece datele sunt imagini, analiza statistică clasică (medie, mediană) se va aplica pe intensitatea pixelilor.*

### 2.1 Statistici planificate
* **Analiza distribuției claselor:** Verificarea echilibrului între clasa "OK" și "Defect" (țintă: 50%-50% sau balansat).
* **Histograma intensității pixelilor:** Analiza diferențelor de luminozitate medie între imaginile cu defecte (ex: mai întunecate pentru "lack of fusion") și cele normale.

### 2.2 Analiza calității
* **Verificare vizuală:** Inspectarea randomizată a 10 imagini din fiecare clasă pentru a valida corectitudinea simulării.
* **Dimensiuni:** Verificarea consistenței rezoluției (toate imaginile trebuie să aibă strict 64x64 px).

## 3. Preprocesarea Datelor (Planificat)

### 3.1 Curățarea datelor
Deoarece datele sunt generate sintetic, riscul de date lipsă (NaN) este zero. Totuși, se vor aplica următoarele verificări automate:
* **Verificare dimensiuni:** Eliminarea oricărei imagini care nu are exact 64x64 pixeli.
* **Verificare format:** Conversia forțată la Grayscale (1 canal) dacă imaginea este salvată greșit ca RGB.

### 3.2 Transformarea caracteristicilor
* **Normalizare:** Valorile pixelilor (0–255) vor fi împărțite la 255.0 pentru a obține valori în intervalul `[0, 1]`. Aceasta ajută la convergența mai rapidă a rețelei neuronale (CNN).
* **Encoding:** Etichetele vor fi numerice: `0` pentru OK și `1` pentru Defect.

### 3.3 Structurarea seturilor de date
Setul total de imagini va fi împărțit aleatoriu (folosind un `seed` fix pentru reproductibilitate) în:
* **70% Train:** Pentru antrenarea ponderilor (weights).
* **15% Validation:** Pentru tunarea hiperparametrilor și oprirea antrenării (Early Stopping).
* **15% Test:** Pentru evaluarea finală a performanței.

### 3.4 Salvarea rezultatelor
* Imaginile procesate nu vor fi duplicate fizic pentru a economisi spațiu, ci vor fi procesate "on-the-fly" folosind un generator de date (`tf.data.Dataset` sau `ImageDataGenerator`).

## 4. Fișiere Generate în Această Etapă

* `data/README.md` – Descrierea detaliată a setului de date.
* `src/preprocessing/utils.py` – Scheletul codului de preprocesare.
* `docs/` – Prezentările PowerPoint ale proiectului.

---

## 5. Stare Etapă

- [x] Structură repository configurată
- [x] Dataset analizat (Plan EDA realizat în documentație)
- [ ] Date preprocesate (Urmează generarea acasă/pana la urmatorul laborator)
- [ ] Seturi train/val/test generate (Urmează generarea acasă/pana la urmatorul laborator)
- [x] Documentație actualizată în README + `data/README.md`