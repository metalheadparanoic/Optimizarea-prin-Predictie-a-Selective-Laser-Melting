# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nicolae Tudor-Stefan]  
**Link Repository GitHub:** [https://github.com/metalheadparanoic/Optimizarea-prin-Predictie-a-Selective-Laser-Melting]  
**Data predării:** [11.12.2025]

---

## PREREQUISITE – Verificare Etapa 4

- [x] **State Machine** definit și documentat în `docs/state_machine.*`
- [x] **Contribuție ≥40% date originale** în `data/generated/` 
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Cerințe Nivel 1 – Obligatoriu (Realizat)

Am îndeplinit toate cele 7 puncte obligatorii, adaptate pentru proiectul de monitorizare SLM, după cum urmează:

### 1. Antrenare model pe setul final
- **Status:** [x] Realizat
- **Descriere:** Am antrenat arhitectura CNN definită în Etapa 4 pe un dataset de **2000 de imagini** (100% contribuție originală, generate sintetic pentru a simula *melt-pool* și defecte de tip *spatter*).
- **Comandă rulată:** `python src/neural_network/train.py`

### 2. Epoci și Batch Size
- **Status:** [x] Realizat
- **Configurație:** Antrenamentul a rulat timp de **10 epoci** cu un **batch size de 32**.
- **Observație:** Modelul a convers rapid, atingând o acuratețe ridicată încă din primele epoci datorită calității datelor sintetice (zgomot controlat).

### 3. Împărțire Stratificată (70% / 15% / 15%)
- **Status:** [x] Realizat
- **Implementare:** Împărțirea a fost realizată automat în momentul generării dataset-ului (`src/data_acquisition/generate_dataset.py`) pentru a asigura că fiecare subset (Train, Validation, Test) conține un număr echilibrat de clase OK și DEFECT.
- **Verificare:** Scriptul de verificare confirmă distribuția fișierelor în folderele `data/train`, `data/validation`, `data/test`.

### 4. Tabel Justificare Hiperparametri
| **Hiperparametru** | **Valoare Aleasă** | **Justificare pentru proiectul SLM** |
| :--- | :--- | :--- |
| **Learning rate** | 0.001 (Adam default) | Asigură o convergență stabilă a gradientului fără a oscila în jurul minimului, optim pentru imagini grayscale 64x64. |
| **Batch size** | 32 | Compromis ideal între viteza de execuție și utilizarea memoriei RAM, permițând actualizarea frecventă a greutăților (aprox. 44 pași/epocă). |
| **Epochs** | 10 | Suficiente pentru a atinge o acuratețe >85% fără a intra în overfitting, dat fiind că trăsăturile geometrice (elipse) sunt clare. |
| **Optimizer** | Adam | Ales pentru capacitatea de adaptare automată a ratei de învățare, standardul actual pentru rețele convoluționale (CNN). |
| **Loss Function** | Binary Crossentropy | Problema este strict binară: piesa este fie OK, fie DEFECT. Această funcție penalizează direct clasificările greșite. |
| **Activation** | ReLU (hidden) / Sigmoid (out) | **ReLU** pentru eficiență computațională în straturile Conv2D. **Sigmoid** la ieșire forțează rezultatul în intervalul [0, 1] (probabilitate defect). |

### 5. Metrici calculate pe Test Set
- **Status:** [x] Realizat
- **Rezultate obținute:**
    - **Acuratețe:** ~89% 
    - **Loss:** ~0.25
    - **F1-score:** > 0.85 (Estimare bazată pe matricea de confuzie echilibrată și acuratețea ridicată).

### 6. Salvare model antrenat
- **Status:** [x] Realizat
- **Locație:** Fișierul este salvat în formatul cerut la `models/trained_model.h5`.

### 7. Integrare în UI (Inferență Reală)
- **Status:** [x] Realizat
- **Funcționalitate:** Aplicația web (`src/app/server.py`) a fost actualizată pentru a încărca fișierul `trained_model.h5`. La încărcarea unei imagini noi, sistemul rulează predicția în timp real.
- **Dovada:** Screenshot demonstrativ salvat în `docs/screenshots/inference_real.png`.

**Justificare detaliată pentru Batch Size = 32:**

Am ales `batch_size=32` luând în calcul dimensiunea setului de antrenare ($N_{train} = 1400$ imagini, reprezentând 70% din totalul de 2000).

**Calcul iterații:**
$$\frac{1400 \text{ imagini}}{32 \text{ batch size}} \approx 44 \text{ pași (iterații) per epocă}$$

Această valoare a fost selectată pentru a asigura un echilibru optim între:
1.  **Stabilitatea Gradientului:** Un batch de 32 oferă o medie statistică suficient de relevantă a erorii, reducând zgomotul (oscilațiile) gradientului care ar apărea la batch-uri foarte mici (ex: 4 sau 8), permițând o descendență mai lină către minimul global.
2.  **Eficiența Memoriei:** Deși imaginile sunt mici (64x64 px), procesarea a 32 de imagini simultan este extrem de eficientă pentru memoria RAM/VRAM disponibilă, evitând bottleneck-urile de transfer de date.
3.  **Viteza de Convergență:** Actualizarea ponderilor de **44 de ori pe epocă** a permis modelului să învețe rapid trăsăturile distinctive ale defectelor, atingând o acuratețe ridicată (>85%) în doar 10 epoci, fără a necesita un timp de antrenare excesiv.