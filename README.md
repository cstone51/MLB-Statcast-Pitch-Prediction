# MLB Pitch Prediction

> A machine learning system for predicting pitch type selection and identifying key drivers by using MLB statcast data

---
## Link to Analysis Paper : https://github.com/cstone51/MLB-Statcast-Pitch-Prediction/blob/main/MLB_Pitch_Prediction%20-%20Analysis%20Paper.pdf

## Overview

This project trains a multiclass random forest classifier to predict the next pitch type thrown by a given MLB pitcher based on contextual game state features like count, inning, batter handedness, baserunner configuration, and pitch sequencing history. Models are trained on historical statcast data retrieved via [pybaseball](https://github.com/jldbc/pybaseball) and evaluated using log loss, accuracy, and F1-score against a naive baseline.


---

## Results Summary

| Model | Log Loss | vs. Baseline | Accuracy | F1 (Weighted) | Top-2 Acc |
|---|---|---|---|---|---|
| Random Forest | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| KNN | — | — | — | — | — |
| **Baseline** | — | — | — | — | — |

> Baseline: always predict training set class distribution regardless of game state.

---

## Project Structure

```
MLB Pitch Prediction/
│
├── src/
│   ├── __init__.py          # package exports
│   ├── data.py              # MLB statcast data retrieval from pybaseball and cleaning
│   ├── features.py          # feature engineering and sklearn pipeline
│   ├── train.py             # end-to-end training run with MLflow logging
│   └── predict.py           # loads artifacts and exposes inference interface
│
├── models/
│   ├── model.joblib             # serialized sklearn pipeline (produced by train)
│   └── label_encoder.joblib     # fitted LabelEncoder (produced by train)
│
├── notebooks/
│   └── EDA.ipynb    # scratch space for EDA and experimentation
    └── model.ipynb # scratch space for model testing and validation
│
├── mlruns/                  # MLflow experiment logs 
├── requirements.txt
└── README.md
```

---

## Features

| Feature | Type | Description |
|---|---|---|
| `balls` | Numeric | Current ball count |
| `strikes` | Numeric | Current strike count |
| `inning` | Numeric | Inning number |
| `outs_when_up` | Numeric | Outs in the inning |
| `inning_top` | Categorical | Top (1) or bottom (0) of inning |
| `batter_is_right` | Categorical | Batter handedness — right (1) or left (0) |
| `runner_on_first` | Categorical | Runner on first base (1/0) |
| `prev_pitch_1` | Categorical | Pitch type thrown immediately prior — scoped to current at-bat |
| `prev_pitch_2` | Categorical | Pitch type thrown two pitches prior — scoped to current at-bat |

> Previous pitch features use `groupby(at_bat_number)` before shifting to prevent cross-at-bat leakage.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```python
from src.train import train

train(
    pitcher_first="Chris",
    pitcher_last="Sale",
    start_date="2023-01-01",
    end_date="2025-12-31"
)
```

This will:
- Pull and clean Statcast data for the specified pitcher and date range
- Engineer features and apply a chronological 65/35 train/test split
- Train the configured classifier and log all metrics to MLflow
- Save `model.joblib` and `label_encoder.joblib` to `models/`

### 3. Make a prediction

```python
from src.predict import predict_pitch

result = predict_pitch({
    "balls":           1,
    "strikes":         2,
    "inning":          6,
    "outs_when_up":    1,
    "inning_top":      1,
    "batter_is_right": 1,
    "runner_on_first": 0,
    "prev_pitch_1":    "FF",
    "prev_pitch_2":    "FF",
})

# output:
# {
#     "prediction": "SL",
#     "probabilities": {"SL": 0.41, "FF": 0.31, "CH": 0.18, "CU": 0.10}
# }
```

### 4. View experiment runs

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to compare runs across pitchers and hyperparameter configurations.

---

## Methodology

### Train / Test Split
a **chronological split** is used (first 65% of pitches = train, last 35% = test) rather than a random shuffle. This reflects real-world deployment conditions — the model is always predicting future pitches from a pitcher it has only observed historically and prevents future at-bat data from leaking into the training set.

### Minority Class Handling
pitch types appearing fewer than a configurable threshold of times are removed before training to prevent `stratify` errors and ensure the model is not evaluated on classes it had insufficient signal to learn. Test rows containing pitch types unseen during training are filtered out separately after the split.

### Class Weighting
Rather than using raw class frequencies (which causes minority pitch types to never be predicted) or fully balanced weights (which distorts the natural pitch distribution), a **square root inverse frequency** weighting scheme is applied. This nudges the model toward minority classes while preserving the pitcher's natural pitch mix tendencies.

### Evaluation
All models are compared against a **naive baseline** that always predicts the training set class distribution. Log loss improvement over this baseline is the primary evaluation metric, as it captures both classification accuracy and probability calibration 

---


---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
pybaseball
mlflow
joblib
python-dateutil
```



## Acknowledgements

Pitch data sourced via [pybaseball](https://github.com/jldbc/pybaseball), which provides a Python interface to Baseball Savant's Statcast database.


