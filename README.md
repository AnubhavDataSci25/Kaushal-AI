# Kaushal AI – Career Recommendation System

**Kaushal AI** is an end-to-end machine-learning web application that recommends a suitable career path to a user based on their education, years of experience, technical skills, area of interest, certifications, project experience, and learning background.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
  - [Data Ingestion](#data-ingestion)
  - [Data Transformation](#data-transformation)
  - [Model Training](#model-training)
  - [Prediction Pipeline](#prediction-pipeline)
- [Web Application](#web-application)
- [Demo App (Streamlit)](#demo-app-streamlit)
- [Unit Tests](#unit-tests)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Project Internals](#project-internals)

---

## Project Overview

Kaushal AI takes a user's profile inputs—education level, years of experience, skills, area of interest, certifications, number of projects, learning source, and dominant project domain—and returns the most suitable career recommendation. The underlying model is trained on a **20 000-row** synthetic career dataset and is selected automatically through a multi-model comparison step.

---

## Features

- **Multi-model training & auto-selection** – trains 9 classifiers and picks the best-performing one.
- **Custom preprocessing pipeline** – handles one-hot encoding, multi-label binarization (skills), and label encoding.
- **Expanded feature set** – now includes `projects_count`, `learning_source`, and `dominant_project_domain` in addition to the core profile fields.
- **Flask web interface** – polished, responsive UI with a modern navbar, hero section, loading overlay, and result card.
- **Streamlit demo app** – `demo_app.py` provides an interactive prototype using the full expanded feature set.
- **Modular ML codebase** – distinct components for ingestion, transformation, training, and inference with structured logging and custom exceptions.
- **Artifact persistence** – trained model, preprocessor, and target encoder are saved as `.pkl` files under `artifacts/`.
- **Unit tests** – pytest suite covering core utilities (`evaluate_models`, `save_objects`, `load_objects`).

---

## Project Structure

```
Kaushal-AI/
├── app.py                        # Flask application entry point
├── demo_app.py                   # Streamlit demo app (expanded feature set)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup (CareerLens AI)
├── pytest.ini                    # pytest configuration
│
├── artifacts/                    # Generated ML artifacts
│   ├── data.csv                  # Raw dataset copy
│   ├── train.csv / test.csv      # Train/test splits
│   ├── model.pkl                 # Best trained model
│   ├── preprocessor.pkl          # Fitted preprocessing pipeline
│   └── target_encoder.pkl        # Fitted LabelEncoder for target
│
├── notebook/
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   ├── MODEL_TRAINING.ipynb      # Model training experiments
│   ├── demo_artifacts/           # Pre-trained artifacts for the Streamlit demo
│   │   ├── rf_model.pkl
│   │   ├── ohe_encoder.pkl
│   │   ├── mlb_encoder.pkl
│   │   └── target_career_le_encoder.pkl
│   └── data/
│       ├── career_dataset_10k.csv            # Original 10k-row dataset
│       ├── career_dataset_20k.csv            # Upgraded 20k-row dataset
│       ├── career_dataset_20k_with_meta.csv  # 20k dataset with metadata columns
│       ├── cleaned_career_dataset.csv        # Cleaned 10k dataset
│       └── cleaned_career_dataset_20k.csv    # Cleaned 20k dataset
│
├── src/
│   ├── execption.py              # CustomException with traceback details
│   ├── utils.py                  # save_objects, load_objects, evaluate_models
│   ├── logger/
│   │   └── loggings.py           # Logging configuration
│   ├── components/
│   │   ├── data_ingestion.py     # Reads raw CSV, saves train/test splits
│   │   ├── data_transformation.py# Feature engineering & preprocessing
│   │   └── model_trainer.py      # Trains models, selects best, saves artifact
│   └── pipeline/
│       └── predict_pipeline.py   # PredictPipeline & CustomData classes
│
├── tests/
│   └── test_utils.py             # Unit tests for utility functions
│
└── templates/
    └── index.html                # Jinja2 HTML template (Bootstrap 5 UI)
```

---

## Dataset

| Property | Detail |
|---|---|
| File | `notebook/data/career_dataset_20k.csv` |
| Rows | ~20 000 |
| Key columns | `education`, `experience_years`, `skills`, `interests`, `certification`, `projects_count`, `learning_source`, `dominant_project_domain`, `target_career` |

The `skills` column contains comma-separated lists. `interests` may be a single string or a list-like string. Three new columns were introduced in the 20k dataset upgrade: `projects_count` (numeric), `learning_source` (categorical), and `dominant_project_domain` (categorical).

---

## ML Pipeline

### Data Ingestion

`src/components/data_ingestion.py`

- Reads the raw CSV (`career_dataset_20k.csv`) from `notebook/data/`.
- Saves a copy to `artifacts/data.csv`.
- Splits data 75 / 25 into `artifacts/train.csv` and `artifacts/test.csv`.

### Data Transformation

`src/components/data_transformation.py`

Three parallel transformation branches are assembled via `ColumnTransformer`:

| Branch | Columns | Transformer |
|---|---|---|
| Numeric | `experience_years`, `projects_count` | `SimpleImputer(median)` |
| Categorical OHE | `education`, `interests`, `certification`, `learning_source`, `dominant_project_domain` | `SimpleImputer` + `OneHotEncoder` |
| Skills MLB | `skills` | Custom `MultiLabelBinarizerTransformer` |

The target column `target_career` is label-encoded with `LabelEncoder`.  
Both the preprocessor and target encoder are saved to `artifacts/`.

### Model Training

`src/components/model_trainer.py`

All 9 classifiers below are trained and evaluated:

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| Decision Tree | scikit-learn |
| AdaBoost | scikit-learn |
| Gradient Boosting | scikit-learn |
| K-Nearest Neighbors | scikit-learn |
| Support Vector Machine | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |

The model with the highest test-set accuracy (minimum threshold: **0.70**) is saved to `artifacts/model.pkl`.

### Prediction Pipeline

`src/pipeline/predict_pipeline.py`

- `PredictPipeline.predict(features)` – loads artifacts, applies the preprocessor, runs inference, and inverse-transforms labels back to human-readable career names.
- `CustomData` – converts raw user inputs into a pandas DataFrame that matches the training schema.

---

## Web Application

`app.py` — Flask server running on port **5000**.

The frontend was redesigned with a polished modern UI featuring a navbar, hero section, loading overlay, and result card.

### Input fields

| Field | Type | Options |
|---|---|---|
| Education | Dropdown | BCA, BSc, BTech, Diploma, MBA, MCA |
| Years of Experience | Range slider | 0 – 15 |
| Area of Interest | Dropdown | AI, Business, Data Science, Design, Marketing, Cybersecurity, Web Dev |
| Certification | Dropdown | None, AWS, Azure, Coursera ML, Google Data Analytics, Udemy Web Dev |
| Skills | Multi-select chips | 28 skills (Python, SQL, ML, React, etc.) |

### Output

The predicted career role is displayed as a highlighted result card on the same page after form submission.

---

## Demo App (Streamlit)

`demo_app.py` — Interactive Streamlit prototype using the expanded 20k-dataset feature set.

Run with:

```bash
streamlit run demo_app.py
```

### Additional input fields (vs. Flask app)

| Field | Type | Options |
|---|---|---|
| Number of Projects | Slider | 0 – 10 |
| Learning Source | Dropdown | bootcamp, online-course, self-taught, university, workplace |
| Dominant Project Domain | Dropdown | none, business, cloud, data, design, devops, marketing, ml, mobile, security, web |

The demo uses dedicated pre-trained artifacts stored under `notebook/demo_artifacts/` (Random Forest model, OHE encoder, MLB encoder, and target label encoder).

---

## Unit Tests

Tests are located in `tests/test_utils.py` and cover the core utility functions:

- `test_evaluate_models_returns_report` – verifies that `evaluate_models` returns a dict with valid accuracy scores.
- `test_save_and_load_objects` – round-trip test for `save_objects` / `load_objects`.

Run the suite with:

```bash
pytest
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Flask |
| Demo UI | Streamlit |
| Frontend | HTML5, Bootstrap 5, Bootstrap Icons |
| ML / Data | scikit-learn, XGBoost, LightGBM, pandas, NumPy |
| Serialization | joblib |
| Testing | pytest |
| Notebooks | Jupyter (EDA & model experiments) |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/AnubhavDataSci25/Kaushal-AI.git
cd Kaushal-AI

# Install dependencies (also installs the package in editable mode)
pip install -r requirements.txt
```

### Running the App

**Option 1 – Use pre-built artifacts (recommended)**

The `artifacts/` folder already contains a trained model and preprocessors. Simply run:

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

**Option 2 – Retrain from scratch**

```bash
python src/components/data_ingestion.py
```

This re-runs the full pipeline (ingestion → transformation → training) and overwrites the artifacts, then you can start the app as above.

**Option 3 – Run the Streamlit demo**

```bash
streamlit run demo_app.py
```

---

## Project Internals

- **Logging** – all pipeline stages write timestamped logs via `src/logger/loggings.py`.
- **Custom exceptions** – `src/execption.py` wraps every exception with the script name and line number for easy debugging.
- **`src/utils.py`** – shared helpers: `save_objects`, `load_objects`, and `evaluate_models`.
- **`setup.py`** – registers the project as an installable package named *CareerLens AI* so `src` imports work without path manipulation.
