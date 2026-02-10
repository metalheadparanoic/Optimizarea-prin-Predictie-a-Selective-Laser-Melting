# Modul 3: Interfață Web și Serviciu de Predicție

Acest modul implementează interfața cu utilizatorul (Frontend) și logica de servire a modelului (Backend de inferență), simulând un mediu de producție industrială (Industry 4.0).

Aplicația este construită folosind **Flask** (Python Microframework) și oferă operatorilor o metodă simplă și vizuală de a verifica calitatea procesului SLM pe baza imaginilor termice, fără a interacționa direct cu codul sursă sau scripturile de antrenare.

## Descriere Funcțională

Modulul transformă modelul de inteligență artificială antrenat (din Modulul 2) într-un serviciu web accesibil.

### Funcționalități Cheie:
1.  **Încărcare Dinamică a Modelului:** Aplicația caută automat cel mai performant model disponibil. Prioritizează `models/optimized_model.h5` (rezultat din Grid Search); dacă acesta lipsește, încarcă `models/trained_model.h5` (baseline).
2.  **Preprocesare Automată:** Imaginile încărcate de utilizator sunt convertite automat la formatul necesar rețelei neuronale (Grayscale, redimensionare 64x64 pixeli, normalizare), transparent pentru utilizator.
3.  **Feedback Vizual Instant:** Rezultatele sunt afișate prin coduri de culoare:
    * **VERDE:** Clasa `OK` (Proces stabil).
    * **ROȘU:** Clasa `DEFECT` (Instabilitate detectată).
4.  **Audit și Logging:** Fiecare predicție este salvată automat într-un fișier CSV (`results/production_log.csv`) pentru trasabilitate și controlul calității (QA).

## Arhitectura Fișierului `main.py`

Fișierul `main.py` conține întreaga logică a serverului web:

* **Configurare:** Definește căile către modele și folderul de rezultate.
* **Rute Flask:** Gestionează cererile HTTP (`GET` pentru afișarea interfeței, `POST` pentru upload-ul imaginilor).
* **Motor de Inferență:** Funcția care preia imaginea, o trece prin rețeaua neuronală și interpretează probabilitatea (Threshold implicit: 0.5).
* **Interfață HTML:** Codul HTML/Bootstrap este integrat direct pentru simplitate și portabilitate, generând pagina web de interacțiune.

## Instrucțiuni de Utilizare

Pentru a lansa aplicația în modul de producție/demonstrație, urmați pașii:

### 1. Precondiții
Asigurați-vă că ați rulat deja scripturile de antrenare/optimizare și că există cel puțin un fișier `.h5` valid în folderul `models/`.

### 2. Lansare Server
Din rădăcina proiectului, executați comanda:

```bash
python src/app/main.py
```
Veți vedea un mesaj de confirmare în consolă: `[INFO] Pornire Server Flask...`  
`[INFO] Accesati in browser: http://127.0.0.1:5000`

### 3. Utilizare Interfață

1.  Deschideți browserul web la adresa [http://127.0.0.1:5000].
2.  Apăsați butonul "**Choose File**" și selectați o imagine termică (din `data/test/` sau o imagine nouă).
3.  Apăsați "**Analizează Imaginea**".
4.  Vizualizați rezultatul și scorul de încredere.

## Structura Log-urilor

Rezultatele sunt salvate în `results/production_log.csv` sub formatul:

```csv
timestamp, filename, raw_score, label, model_version