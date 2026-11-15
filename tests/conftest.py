import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mlflow_tmp(tmp_path):
    """Point MLflow at a throwaway SQLite DB + artifact folder so tests never touch ./mlflow.db."""
    import mlflow

    from lab.tracking import configure

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    configure(uri, experiment_name="test-experiment",
              artifact_location=(tmp_path / "artifacts").as_uri())
    yield uri
    if mlflow.active_run():
        mlflow.end_run()
