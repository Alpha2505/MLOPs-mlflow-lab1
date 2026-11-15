"""Hyperparameter sweep with nested runs; registers the best model and tags it 'champion'.

Example:
    python -m lab.sweep --metric recall
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

from lab.data import load_dataset
from lab.tracking import CHAMPION_ALIAS, MODEL_NAME, configure
from lab.train import train_and_log

DEFAULT_GRID = [
    ("logreg", {"C": 0.1}),
    ("logreg", {"C": 1.0}),
    ("logreg", {"C": 10.0}),
    ("random_forest", {"n_estimators": 100, "max_depth": 4}),
    ("random_forest", {"n_estimators": 300, "max_depth": 8}),
    ("grad_boost", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2}),
    ("grad_boost", {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3}),
]
THRESHOLDS = [0.3, 0.5]
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def pick_best(results: list[dict], metric: str) -> dict:
    """Highest `metric`, ties broken by roc_auc so the choice is deterministic."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    return max(results, key=lambda r: (r["metrics"][metric], r["metrics"]["roc_auc"]))


def run_sweep(grid=None, thresholds=None, metric: str = "f1") -> dict:
    """Train every (model, params, threshold) combination as a child run of one parent run."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    grid = grid or DEFAULT_GRID
    thresholds = thresholds or THRESHOLDS
    data = load_dataset()

    results = []
    with mlflow.start_run(run_name=f"sweep-by-{metric}") as parent:
        mlflow.log_param("selection_metric", metric)
        mlflow.log_param("n_candidates", len(grid) * len(thresholds))
        for model_name, params in grid:
            for t in thresholds:
                name = f"{model_name}-" + "-".join(f"{k}{v}" for k, v in params.items()) + f"-t{t}"
                res = train_and_log(model_name, params, data, threshold=t,
                                    nested=True, run_name=name)
                results.append({**res, "model_name": model_name, "params": params, "threshold": t})

        best = pick_best(results, metric)
        mlflow.log_metric(f"best_{metric}", best["metrics"][metric])
        mlflow.set_tag("best_child_run_id", best["run_id"])
        parent_id = parent.info.run_id

    version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    return {**best, "parent_run_id": parent_id, "model_version": version.version}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Sweep models/thresholds and register the best")
    p.add_argument("--metric", choices=METRICS, default="f1")
    p.add_argument("--tracking-uri", default=None)
    args = p.parse_args(argv)

    configure(args.tracking_uri)
    best = run_sweep(metric=args.metric)
    print(f"Best by {args.metric}: {best['model_name']} {best['params']} threshold={best['threshold']}")
    print(f"  {args.metric} = {best['metrics'][args.metric]:.4f}, roc_auc = {best['metrics']['roc_auc']:.4f}")
    print(f"Registered {MODEL_NAME} v{best['model_version']} as @{CHAMPION_ALIAS}")


if __name__ == "__main__":
    main()
