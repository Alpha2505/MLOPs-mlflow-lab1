"""MLflow setup shared by training, sweep and prediction."""
import os

import mlflow

# A SQLite backend (instead of the default ./mlruns file store) enables the Model Registry.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_NAME = "BreastCancerClassifier"
CHAMPION_ALIAS = "champion"


def configure(tracking_uri: str | None = None,
              experiment_name: str = EXPERIMENT_NAME,
              artifact_location: str | None = None) -> str:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    return uri
