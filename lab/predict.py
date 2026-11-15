"""Load the @champion model from the registry and predict, using the threshold it was tuned with.

Example:
    python -m lab.predict --n 10
"""
import argparse

import mlflow.sklearn
from mlflow.tracking import MlflowClient

from lab.data import load_dataset
from lab.metrics import apply_threshold
from lab.tracking import CHAMPION_ALIAS, MODEL_NAME, configure


def load_champion():
    """Return (sklearn model, threshold, version) for the version aliased 'champion'."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    threshold = float(client.get_run(mv.run_id).data.params.get("threshold", 0.5))
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    return model, threshold, mv.version


def predict(model, X, threshold: float):
    proba = model.predict_proba(X)[:, 1]
    return apply_threshold(proba, threshold), proba


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Predict with the champion model")
    p.add_argument("--n", type=int, default=5, help="number of test rows to score")
    p.add_argument("--tracking-uri", default=None)
    args = p.parse_args(argv)

    configure(args.tracking_uri)
    model, threshold, version = load_champion()
    data = load_dataset()
    X, y = data.X_test.head(args.n), data.y_test.head(args.n)
    preds, proba = predict(model, X, threshold)

    print(f"{MODEL_NAME} v{version} (@{CHAMPION_ALIAS}), threshold={threshold}")
    print(f"{'row':>4} {'p(malignant)':>13} {'pred':>5} {'actual':>7}")
    for i, (pr, label, actual) in enumerate(zip(proba, preds, y)):
        print(f"{i:>4} {pr:>13.3f} {label:>5} {actual:>7}")


if __name__ == "__main__":
    main()
