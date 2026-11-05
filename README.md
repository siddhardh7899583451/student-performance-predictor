---
title: Student Performance Predictor
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
tags:
  - machine-learning
  - linear-regression
  - education
  - scikit-learn
---

# 📚 Student Performance Predictor

Educational **Multiple Linear Regression** lab built with pure Python.

Predict a student’s exam score from study hours, attendance, previous score, sleep, and assignments completed — then inspect metrics, residuals, and coefficient explanations.

Designed as a clean, portfolio-ready student project that maps directly onto standard supervised-learning syllabus topics.

---

## What this project covers

| Syllabus topic | Implementation |
|----------------|----------------|
| Multiple Linear Regression | Default model; \(\hat{y} = \beta_0 + \sum_j \beta_j x_j\) |
| Ridge Regression (L2) | Toggle + α slider; coefficient shrinkage |
| Train / test split | Configurable test size; metrics reported separately |
| No data leakage | `StandardScaler` fitted **only** on the training fold |
| Feature scaling | Optional StandardScaler so coefficients are comparable |
| Polynomial features | Degree 1 or 2 (simple bias–variance experiment) |
| Evaluation metrics | R², MAE, RMSE + 5-fold Cross-Validation R² |
| Residual analysis | Actual-vs-predicted + residual scatter plots |
| Explainability | Sorted coefficient table + plain-English interpretation |
| Interactive inference | What-if predictor with contribution breakdown |

---

## Project layout

```
student-performance-predictor/
├── README.md                 # This file (also HF Spaces config)
├── requirements.txt
├── .gitignore
├── app.py                    # Gradio web app (entry point)
├── ml_engine.py              # Core training / evaluation / prediction logic
├── generate_data.py          # Reproducible synthetic dataset generator
├── data/
│   └── student_performance.csv
├── tests/
│   └── test_engine.py
└── scripts/                  # (optional helpers)
```

No placeholders. No TODOs. Ready to run and ready to push.

---

## Quick start (local)

```bash
# 1. Clone / download and enter the folder
cd student-performance-predictor

# 2. (Recommended) virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Gradio app
python app.py
```

Open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

### Regenerate the dataset (optional)

```bash
python generate_data.py
```

### Run unit tests

```bash
pytest -v
```

Expected: all tests pass.

---

## How to use the web app

1. **Data Overview** – inspect the 400-row synthetic dataset and feature correlations.
2. **Train & Evaluate** – choose Linear or Ridge, toggle scaling / polynomial degree, train, then read R² / MAE / RMSE / CV and the coefficient explanations. Residual plots appear automatically.
3. **What-If Predictor** – move the sliders for a hypothetical student and get a predicted exam score plus approximate feature contributions.
4. **Concepts** – syllabus map and suggested experiments.

---

## Tech stack

- **Python 3.10+**
- **scikit-learn** – LinearRegression, Ridge, train_test_split, StandardScaler, PolynomialFeatures, cross_val_score, metrics
- **pandas / NumPy** – data handling
- **matplotlib** – residual & coefficient plots
- **Gradio** – interactive UI (local + Hugging Face Spaces)
- **pytest** – unit tests

---

## Hugging Face Spaces

This repository is already configured for Spaces:

- Front-matter YAML at the top of `README.md`
- `app_file: app.py`
- `sdk: gradio`

Push the repo to a new Space (or connect an existing GitHub repo) and the app will build automatically.

---

## Design notes (why it is structured this way)

- **Single responsibility modules** – `ml_engine.py` owns all modelling logic; `app.py` only wires the UI.
- **No leakage** – scaler and polynomial transformer are fit exclusively on the training split.
- **Explainable by default** – every trained model returns a coefficient table with human-readable meaning.
- **Reproducible data** – `generate_data.py` uses a fixed seed and a transparent linear data-generating process.
- **Tested** – core training, prediction, and polynomial expansion paths are covered by pytest.
- **Python-first** – minimal dependencies, no heavy web framework beyond Gradio.

---

## License & intent

Created as a clear, learnable student project for practicing supervised regression, evaluation, and basic model interpretation.

Feel free to fork, extend (add more features, real CSV upload, SHAP, etc.), or use as a template for other regression labs.
