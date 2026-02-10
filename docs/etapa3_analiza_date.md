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

## 2. Analiza Exploratorie a Datelor (EDA)

*Deoarece datele sunt imagini, analiza statistică clasică (medie, mediană) se va aplica pe intensitatea pixelilor.*

### 2.1 Statistici planificate
* **Analiza distribuției claselor:** Verificarea echilibrului între clasa "OK" și "Defect" (țintă: 50%-50% sau balansat).
* **Histograma intensității pixelilor:** Analiza diferențelor de luminozitate medie între imaginile cu defecte (ex: mai întunecate pentru "lack of fusion") și cele normale.

### 2.2 Analiza calității
* **Verificare vizuală:** Inspectarea randomizată a 10 imagini din fiecare clasă pentru a valida corectitudinea simulării.
* **Dimensiuni:** Verificarea consistenței rezoluției (toate imaginile trebuie să aibă strict 64x64 px).

## 3. Preprocesarea Datelor

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
- [x] Date preprocesate (Generat local 2000 imagini sintetice SLM)
- [x] Seturi train/val/test generate (Split automat: 70% Train, 15% Val, 15% Test)
- [x] Documentație actualizată în README + `data/README.md`

## 6. Tabelul Nevoie Reală → Soluție SIA → Modul Software 

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul nostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Reducerea costurilor de producție SLM prin detectarea instabilităților de proces | Clasificare imagini melt-pool (OK/Defect) în timp real → alertă operator | `src/neural_network` (Model CNN)|
| Eliminarea inspecției post-procesare distructive sau CT | Validare calitate strat-cu-strat cu acuratețe > 89% și timp de răspuns sub 1s |`src/app` (Web Service UI) |
| Generarea de date de antrenament pentru defecte rare (lipsă fuziune, spatter) | Simulare fizică a geometriei melt-pool-ului și a zgomotului de senzor | `src/data_acquisition` |

## 7. Contribuția originală la setul de date

**Total observații finale:** 2000
**Observații originale:** 2000 (100%)

**Tipul contribuției:**
[X] Date generate prin simulare fizică
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală
[ ] Date sintetice prin metode avansate

**Descriere detaliată:**
Am dezvoltat un generator propriu în Python (`src/data_acquisition/generate_dataset.py`) care realizează o **simulare geometrică a fenomenelor fizice** din procesul SLM, așa cum sunt ele capturate de senzorii optici.

Abordarea nu este o simulare termică (CFD), ci o **modelare a semnăturii vizuale** bazată pe comportamentul fizic cunoscut al materialului, necesară, deoarece accesul la o mașină SLM industrială cu monitorizare in-situ nu a fost posibil:
1.  **Tensiunea Superficială (Clasa OK):** Simulată prin generarea de forme cu excentricitate redusă (cvasicirculare), care imită tendința metalului lichid de a minimiza suprafața.
2.  **Instabilitatea Hidrodinamică (Clasa DEFECT):** Simulată prin alungirea geometrică a bazinului (fenomenul de *keyhole collapse* sau viteză prea mare) și generarea stocastică de particule satelit (fenomenul de *spatter* cauzat de ejecția de material).
3.  **Modelul Senzorului:** S-a aplicat filtrare Gaussiană pentru a simula emiterea difuză a radiației termice ("glow") și zgomot aditiv pentru a reproduce caracteristicile unei camere CCD industriale.

**Locația codului:** `src/data_acquisition/generate_dataset.py`
**Locația datelor:** `data/raw/` și `data/processed/`

**Dovezi:**
- Scriptul de generare parametrizabil: `src/data_acquisition/generate_dataset.py`
- Structura dataset-ului balansat (1000 OK / 1000 Defect) vizibilă în folderul `data/`.

## 8. Diagrama State Machine a Întregului Sistem

**Diagrama vizuală:** `docs/state_machine.png`

#### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **B. Clasificare imagini defecte producție**, deoarece SIA-ul funcționează ca un sistem de inspecție "Pas cu Pas" (Layer-wise). La fiecare nou strat depus de mașina SLM, sistemul este declanșat (Trigger), capturează imaginea melt-pool-ului, o validează și o clasifică pentru a decide dacă procesul poate continua la stratul următor sau trebuie oprit.

**Stările principale sunt:**
1. **[IDLE]:** Sistemul este în așteptare, monitorizând statusul mașinii SLM.
2. **[WAIT_LAYER_TRIGGER]:** Se așteaptă semnalul că noul strat de pudră a fost depus și laserul este activ.
3. **[CAPTURE_IMAGE]:** Senzorul optic preia instantaneul zonei de topire.
4. **[PREPROCESS]:** Imaginea este validată (să nu fie neagră/blurată), transformată în Grayscale și redimensionată (64x64).
5. **[RN_INFERENCE]:** Modelul CNN procesează tensorul imaginii.
6. **[CLASSIFY_DEFECT]:** Se compară scorul returnat cu pragul de 0.5.
   - **OK:** Loghează succesul -> Se întoarce la WAIT_LAYER_TRIGGER.
   - **DEFECT:** Loghează eroarea -> Declanșează semnalul de STOP / REJECT.

**Tranzițiile critice sunt:**
- **[IDLE] → [WAIT_LAYER]:** Start proces de printare.
- **[CLASSIFY] → [STOP]:** Tranziție critică de siguranță. Dacă detectăm "spatter" excesiv sau deformare, oprim laserul pentru a nu compromite piesa.
- **[CAPTURE] → [ERROR]:** Gestionarea cazurilor în care camera nu răspunde (timeout).

## 9. Scheletul Complet al celor 3 Module

Sistemul este compus din 3 module software independente, interconectate prin fluxul de date, respectând arhitectura SIA propusă:

| **Modul** | **Tehnologie / Locație** | **Stare Funcțională** |
| :--- | :--- | :--- |
| **1. Data Acquisition** | Python / `src/data_acquisition/` | **Funcțional.** Generează dataset sintetic (2000 imagini) simulând fizica procesului SLM. |
| **2. Neural Network** | TensorFlow / `src/neural_network/` | **Funcțional.** Model CNN definit, compilat și antrenat (SavedModel în `models/`). |
| **3. Web Service / UI** | Flask / `src/app/` | **Funcțional.** Interfață web pentru încărcarea imaginilor și vizualizarea diagnosticului. |

### Detalii per modul:

**Modulul 1: Data Logging / Acquisition**
- **Cod:** `src/data_acquisition/generate_dataset.py`
- **Funcționalitate:** Scriptul rulează automat și generează structura de foldere (`data/raw`, `data/train`, etc.).
- **Output:** Produce fișiere `.png` reprezentând melt-pool-ul, împărțite în clasele "ok" și "defect". Imaginile sunt generate procedural folosind OpenCV (elipse, zgomot Gaussian, spatter).

**Modulul 2: Neural Network Module**
- **Cod:** `src/neural_network/model.py` (Definiție) și `train.py` (Antrenare).
- **Arhitectură:** Rețea Neuronală Convoluțională (CNN) cu 3 straturi de convoluție (`Conv2D` + `MaxPooling`) și un clasificator dens (`Dense`).
- **Status:** Modelul este definit, compilat și salvat în formatul `models/slm_model.keras`.

**Modulul 3: Web Service / UI**
- **Cod:** `src/app/server.py`
- **Funcționalitate:** Server web bazat pe **Flask**. Oferă o interfață simplă (HTML/CSS) unde operatorul poate încărca o imagine a stratului curent.
- **Flux:** Imaginea este preprocesată (Grayscale, Resize 64x64), trecută prin modelul din `models/` și rezultatul este afișat colorat (Roșu pentru Defect, Verde pentru OK) împreună cu scorul de încredere.

---

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (Secțiunea 1 din README)
- [x] Declarație contribuție date originale (Secțiunea 2 - 100% original)
- [x] Cod generare date funcțional și documentat
- [x] Dovezi contribuție originală: scriptul `generate_dataset.py` și structura `data/`
- [x] Diagrama State Machine creată și salvată în `docs/state_machine.png`
- [x] Legendă State Machine scrisă în README (Secțiunea 3)
- [x] Repository structurat conform modelului (verificat folderele `models`, `src/app`, `data/generated`)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/data_acquisition/generate_dataset.py`)
- [x] Produce date originale (2000 imagini)
- [x] Datele sunt compatibile cu preprocesarea (format PNG 64x64)

### Modul 2: Neural Network
- [x] Arhitectură RN definită în `src/neural_network/model.py`
- [x] Modelul poate fi salvat și reîncărcat (`models/slm_model.keras`)

### Modul 3: Web Service / UI
- [x] Interfața pornește fără erori (`python src/app/server.py`)
- [x] Primește input de la user (Upload) și afișează output (Predicție)

PROIECT/
├── config/                 <-- (NOU)
│   └── .gitkeep
├── data/
│   ├── generated/          <-- (NOU)
│   ├── processed/
│   ├── raw/ 
│   ├── train/
│   ├── validation/
│   └── test/
├── docs/
│   ├── datasets/
│   |   └── README.md       <-- (Actualizat pentru Etapa 4; Etapa 3 deja existentă)
│   ├── screenshots/        <-- (NOU)
│   ├── state_machine.png   <-- (Diagrama desenată)
│   ├── training_results.png
│   └── ... (PPT-urile vechi)
├── models/
│   └── slm_model.keras     <-- (Modelul mutat aici)
├── src/
│   ├── app/
│   │   └── server.py
│   ├── data_acquisition/
│   │   └── generate_dataset.py
│   ├── neural_network/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   └── preprocessing/
│       ├── process_data.py
│       └── utils.py
├── README.md
├── requirements.txt        <-- (NOU)
├── README_Etapa4_Arhitectura_SIA.md
└── .gitignore