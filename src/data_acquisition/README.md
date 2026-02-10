# Modul 1: Achiziție și Generare Date

Acest modul este responsabil pentru crearea datelor primare necesare antrenării rețelei neuronale. Deoarece datele industriale reale etichetate (imagini melt pool) sunt dificil de obținut și adesea confidențiale, acest proiect utilizează o abordare bazată pe **generare procedurală de date sintetice**.

## Scop

Scopul scripturilor din acest modul este să simuleze fenomene fizice vizibile în procesul de Selective Laser Melting (SLM) și să producă imagini termice (grayscale) care imită semnătura unui bazin de topire (melt pool), atât în stări stabile (OK), cât și instabile (Defect).

## Fișiere Componente

### `generate_dataset.py`

Acesta este scriptul principal de execuție.

* **Funcționalitate:** Generează un număr specificat de imagini pentru fiecare clasă (OK și Defect).
* **Tehnologie:** Utilizează `NumPy` pentru manipularea matricilor de pixeli și `OpenCV` (sau `PIL`) pentru salvarea imaginilor.
* **Output:** Salvează imaginile în folderul `data/raw/` organizate pe subfoldere (`ok/`, `defect/`).

## Algoritmul de Generare

Procesul de simulare se bazează pe următoarele principii fizice simplificate:

### 1. Generarea clasei OK (Stabil)
* **Melt Pool:** Se generează o formă eliptică cu intensitate maximă în centru și scădere graduală (gradient Gaussian) spre margini, simulând distribuția temperaturii.
* **Variabilitate:** Se introduc variații mici, controlate, în dimensiunea axelor elipsei și unghiul de rotație pentru a simula fluctuațiile naturale ale procesului, dar care rămân în limitele de toleranță.
* **Zgomot:** Se adaugă un nivel redus de zgomot Gaussian global pentru a simula granulația senzorului.

### 2. Generarea clasei Defect (Instabil)
Simulează anomalii specifice SLM prin perturbarea parametrilor geometrici și de intensitate:
* **Spatter (Stropi):** Se adaugă puncte de intensitate mare distribuite aleatoriu în jurul bazinului principal, simulând materialul topit expulzat.
* **Porozitate (Keyhole/Lack of Fusion):** Se inserează "găuri" (zone de intensitate mică/negre) în interiorul sau la marginea bazinului, simulând bule de gaz sau lipsa fuziunii.
* **Instabilitate Geometrică:** Se aplică distorsiuni majore ale formei eliptice (alungire excesivă sau formă neregulată) pentru a simula viteze de scanare incorecte sau supraîncălzire.

## Utilizare

Pentru a rula generatorul de date, executați următoarea comandă din rădăcina proiectului:

```bash
python src/data_acquisition/generate_dataset.py
```
Acest script va popula directorul data/raw/ cu imagini noi.

Notă: Rularea acestui script va suprascrie datele brute existente în folderul data/raw.