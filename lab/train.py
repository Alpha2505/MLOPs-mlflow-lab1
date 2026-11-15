"""Train one model and log everything to MLflow.

Example:
    python -m lab.train --model random_forest --n-estimators 300 --max-depth 8 --threshold 0.4
"""
import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from lab.data import Dataset, load_dataset
from lab.metrics import (
    apply_threshold,
    classification_metrics,
    save_confusion_matrix,
    save_roc_curve,
)
from lab.models import MODEL_CHOICES, build_model, resolved_params
from lab.tracking import MODEL_NAME, configure


def train_and_log(model_name: str,
                  params: dict | None,
                  data: Dataset,
                  threshold: float = 0.5,
                  register: bool = False,
                  nested: bool = False,
                  run_name: str | None = None) -> dict:
    """Fit, evaluate and log one model. Returns the run id and test metrics."""
    params = resolved_params(model_name, params)
    model = build_model(model_name, params)

    with mlflow.start_run(run_name=run_name or model_name, nested=nested) as run:
        mlflow.set_tags({"model_type": model_name, "dataset": "sklearn_breast_cancer"})
        mlflow.log_params({
            "model_type": model_name,
            "threshold": threshold,
            "n_train": len(data.X_train),
            "n_test": len(data.X_test),
            **params,
        })

        model.fit(data.X_train, data.y_train)
        y_proba = model.predict_proba(data.X_test)[:, 1]
        y_pred = apply_threshold(y_proba, threshold)

        metrics = classification_metrics(data.y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / "confusion_matrix.png"
            roc_path = Path(tmp) / "roc_curve.png"
            save_confusion_matrix(data.y_test, y_pred, cm_path)
            save_roc_curve(data.y_test, y_proba, roc_path)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        signature = infer_signature(data.X_train, model.predict(data.X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=data.X_train.head(3),
            registered_model_name=MODEL_NAME if register else None,
        )

        return {"run_id": run.info.run_id, "metrics": metrics}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a breast cancer classifier with MLflow tracking")
    p.add_argument("--model", choices=MODEL_CHOICES, default="logreg")
    p.add_argument("--C", type=float, help="logreg: inverse regularization strength")
    p.add_argument("--n-estimators", type=int, help="random_forest / grad_boost")
    p.add_argument("--max-depth", type=int, help="random_forest / grad_boost")
    p.add_argument("--learning-rate", type=float, help="grad_boost")
    p.add_argument("--threshold", type=float, default=0.5, help="probability cut-off for 'malignant'")
    p.add_argument("--register", action="store_true", help="register the model in the Model Registry")
    p.add_argument("--tracking-uri", default=None)
    return p.parse_args(argv)


def cli_params(args: argparse.Namespace) -> dict:
    """Keep only the hyperparameters that apply to the chosen model."""
    allowed = {
        "logreg": {"C"},
        "random_forest": {"n_estimators", "max_depth"},
        "grad_boost": {"n_estimators", "learning_rate", "max_depth"},
    }[args.model]
    given = {"C": args.C, "n_estimators": args.n_estimators,
             "max_depth": args.max_depth, "learning_rate": args.learning_rate}
    return {k: v for k, v in given.items() if k in allowed and v is not None}


def main(argv=None) -> None:
    args = parse_args(argv)
    configure(args.tracking_uri)
    result = train_and_log(args.model, cli_params(args), load_dataset(),
                           threshold=args.threshold, register=args.register)
    print(f"Run {result['run_id']}")
    for name, value in result["metrics"].items():
        print(f"  {name:>9}: {value:.4f}")


if __name__ == "__main__":
    main()
