"""End-to-end checks against a temporary MLflow tracking store."""
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from lab.data import load_dataset
from lab.sweep import pick_best, run_sweep
from lab.tracking import CHAMPION_ALIAS, MODEL_NAME
from lab.train import train_and_log


def test_run_logs_params_metrics_artifacts_and_model(mlflow_tmp):
    data = load_dataset()
    result = train_and_log("logreg", {"C": 0.5}, data, threshold=0.4)

    run = MlflowClient().get_run(result["run_id"])
    assert run.data.params["model_type"] == "logreg"
    assert run.data.params["C"] == "0.5"
    assert run.data.params["threshold"] == "0.4"
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= run.data.metrics[name] <= 1.0

    plots = {a.path for a in MlflowClient().list_artifacts(result["run_id"], "plots")}
    assert plots == {"plots/confusion_matrix.png", "plots/roc_curve.png"}

    model = mlflow.sklearn.load_model(f"runs:/{result['run_id']}/model")
    assert model.predict(data.X_test).shape == (len(data.X_test),)


def test_register_flag_creates_model_version(mlflow_tmp):
    train_and_log("logreg", None, load_dataset(), register=True)
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) == 1


def test_sweep_registers_best_as_champion(mlflow_tmp):
    grid = [("logreg", {"C": 0.1}), ("logreg", {"C": 1.0})]
    best = run_sweep(grid=grid, thresholds=[0.5], metric="f1")

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert mv.run_id == best["run_id"]

    children = client.search_runs(
        [client.get_run(best["parent_run_id"]).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{best['parent_run_id']}'",
    )
    assert len(children) == 2


def test_pick_best_breaks_ties_on_auc():
    results = [
        {"run_id": "a", "metrics": {"f1": 0.9, "roc_auc": 0.95}},
        {"run_id": "b", "metrics": {"f1": 0.9, "roc_auc": 0.99}},
        {"run_id": "c", "metrics": {"f1": 0.8, "roc_auc": 1.00}},
    ]
    assert pick_best(results, "f1")["run_id"] == "b"
