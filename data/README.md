# Dataset: Monitorizare Melt Pool SLM (Sintetic)

Acest director conține datele utilizate pentru antrenarea, validarea și testarea rețelei neuronale convoluționale (CNN) dezvoltate în cadrul proiectului. Imaginile sunt sintetice și simulează semnăturile termice ale bazinului de topire (melt pool) în procesul de Selective Laser Melting (SLM).

## Sumar Dataset

| Caracteristică | Valoare |
| :--- | :--- |
| **Tip Date** | Imagini Termice Simulate (Grayscale) |
| **Format** | PNG |
| **Rezoluție Finală** | 64x64 pixeli |
| **Număr Total Imagini** | ~3000 (variabil în funcție de generare) |
| **Balans Clase** | 50% OK / 50% Defect |
| **Sursă** | Generare procedurală (`src/data_acquisition/generate_dataset.py`) |

---

## Structura Directorului

```text
data/
├── raw/                  # Imaginile brute generate inițial (înainte de split)
│   ├── ok/               # Imagini clasa 'OK'
│   └── defect/           # Imagini clasa 'Defect'
│
├── processed/            # (Opțional) Date intermediare preprocesate
│
├── train/                # Setul de Antrenare (70%)
│   ├── ok/
│   └── defect/
│
├── validation/           # Setul de Validare (15%) - folosit la fiecare epocă
│   ├── ok/
│   └── defect/
│
└── test/                 # Setul de Testare (15%) - neatins până la evaluarea finală
    ├── ok/
    └── defect/
```