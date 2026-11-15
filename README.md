# MLflow Lab 1 — Breast Cancer Classification with Experiment Tracking & Model Registry

This lab uses **MLflow** to track experiments, compare models, register the best one in the **Model Registry**, and serve it as a REST API. It is my own version of MLflow Lab 1 from the MLOps course ([raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps)), rebuilt with a different dataset, a different problem type, and additional logic.

## What I changed from the original lab

| | Original lab | This repo |
|---|---|---|
| **Dataset** | Wine Quality (red wine CSV downloaded from GitHub) and the sklearn diabetes dataset | Wisconsin Breast Cancer (`sklearn.datasets.load_breast_cancer`), 569 samples × 30 features; no download needed |
| **Problem** | Regression (predict a quality score) | Binary classification (malignant vs benign), with malignant as the positive class |
| **Models** | One ElasticNet (plus a RandomForest in the notebook) | Three model families selectable from the CLI: Logistic Regression, Random Forest, Gradient Boosting — each wrapped in a `StandardScaler` pipeline |
| **Metrics** | RMSE, MAE, R² | Accuracy, precision, recall, F1, ROC-AUC |
| **Decision threshold** | n/a | Tunable probability threshold, logged as a parameter (lowering it trades precision for recall, which matters when missing a malignant case is costly) |
| **Artifacts** | Model only | Model + confusion matrix and ROC curve plots |
| **Experiment design** | Single runs launched by hand | Automated sweep: one parent run with **nested child runs** over models × hyperparameters × thresholds |
| **Model Registry** | Skipped when using the file store | SQLite backend so the registry works; best model is registered and given the **`@champion` alias** automatically |
| **Inference** | Requirements-logging demo (`serving.py`) | `predict.py` loads `models:/BreastCancerClassifier@champion` and applies the threshold it was selected with; plus REST serving with `mlflow models serve` |
| **Tests** | None | pytest suite covering data, models, metrics, MLflow logging, registration and the sweep |
| **Structure** | Notebooks + standalone scripts | Python package (`lab/`) with modules for data, models, metrics, tracking, training, sweep and prediction |

## Project structure

```
mlflow-lab1/
├── lab/
│   ├── data.py          # load + stratified split of the breast cancer data
│   ├── models.py        # model factory (3 model types, all sklearn Pipelines)
│   ├── metrics.py       # metrics, threshold logic, confusion matrix / ROC plots
│   ├── tracking.py      # MLflow config (SQLite backend, experiment, model name)
│   ├── train.py         # train + log a single run (CLI)
│   ├── sweep.py         # nested-run sweep, registers best model as @champion
│   └── predict.py       # load @champion from the registry and predict
├── tests/
│   ├── conftest.py      # temp MLflow store fixture
│   ├── test_data.py
│   ├── test_models_and_metrics.py
│   └── test_mlflow_logging.py
├── docs/screenshots/    # MLflow UI screenshots
├── sample_request.json  # example payload for the REST endpoint
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below are run from the repo root.

## How to run

### 1. Start the MLflow UI (in a separate terminal)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://localhost:5001.

### 2. Train single models

```bash
python -m lab.train --model logreg --C 1.0
python -m lab.train --model random_forest --n-estimators 300 --max-depth 8
python -m lab.train --model grad_boost --learning-rate 0.05 --threshold 0.3
```

Each run logs its parameters, the five metrics, the plots under `plots/`, and the model with its signature and an input example. Add `--register` to also register that model.

### 3. Run the sweep and register the best model

```bash
python -m lab.sweep --metric f1        # or: recall, precision, accuracy, roc_auc
```

This trains 14 candidates (7 model configurations × 2 thresholds) as child runs under one parent run, picks the best by the chosen metric (ties broken by ROC-AUC), registers it as `BreastCancerClassifier`, and points the `@champion` alias at the new version.

### 4. Predict with the champion model

```bash
python -m lab.predict --n 10
```

### 5. Serve the champion model as a REST API

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/BreastCancerClassifier@champion" -p 5002 --env-manager local
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:5002/invocations \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

Note: the REST endpoint returns the model's default `predict()` output (0.5 cut-off). `lab/predict.py` applies the tuned threshold.

## Tests

```bash
pytest -v
```

The MLflow tests use a temporary SQLite database and artifact folder, so they never touch your local `mlflow.db`.

## Results

<!-- Fill in after running the sweep -->

| Selection metric | Best model | Params | Threshold | Score | ROC-AUC |
|---|---|---|---|---|---|
| f1 | | | | | |
| recall | | | | | |

## Screenshots

<!-- Add your own screenshots to docs/screenshots/ and update the paths -->

- Experiment list with runs: `docs/screenshots/runs.png`
- Comparing runs (parallel coordinates / metric charts): `docs/screenshots/compare.png`
- Nested sweep runs: `docs/screenshots/sweep.png`
- Model Registry with `@champion` alias: `docs/screenshots/registry.png`
- Logged artifacts (confusion matrix, ROC curve): `docs/screenshots/artifacts.png`
