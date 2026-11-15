import numpy as np
import pytest

from lab.data import load_dataset
from lab.metrics import apply_threshold, classification_metrics
from lab.models import MODEL_CHOICES, build_model, resolved_params


@pytest.mark.parametrize("name", MODEL_CHOICES)
def test_every_model_trains_and_beats_baseline(name):
    data = load_dataset()
    model = build_model(name).fit(data.X_train, data.y_train)
    acc = (model.predict(data.X_test) == data.y_test).mean()
    assert acc > 0.9  # majority-class baseline is ~0.63


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("svm")


def test_overrides_merge_with_defaults():
    assert resolved_params("random_forest", {"max_depth": 3}) == {"n_estimators": 200, "max_depth": 3}


def test_threshold_moves_predictions():
    proba = np.array([0.2, 0.4, 0.6])
    assert apply_threshold(proba, 0.5).tolist() == [0, 0, 1]
    assert apply_threshold(proba, 0.3).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        apply_threshold(proba, 1.5)


def test_metrics_on_known_values():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    y_proba = [0.9, 0.4, 0.3, 0.1]
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["roc_auc"] == 1.0
