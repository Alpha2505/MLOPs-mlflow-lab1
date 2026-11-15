"""Evaluation metrics and plots that get logged to MLflow."""
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def apply_threshold(y_proba, threshold: float = 0.5) -> np.ndarray:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (np.asarray(y_proba) >= threshold).astype(int)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_confusion_matrix(y_true, y_pred, path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["benign", "malignant"],
        ax=ax, colorbar=False,
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
