"""Data loading and splitting for the breast cancer classification lab."""
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET = "malignant"


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_frame() -> pd.DataFrame:
    """Load the Wisconsin breast cancer data as one DataFrame.

    sklearn encodes malignant=0 and benign=1. We flip it so the positive
    class (1) is malignant, which is the case we care about catching.
    Column names get underscores so they are easy to send as JSON when serving.
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df[TARGET] = 1 - df.pop("target")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Dataset:
    """Stratified train/test split so both sets keep the same class balance."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Dataset(X_train, X_test, y_train, y_test)


def load_dataset(test_size: float = 0.2, seed: int = 42) -> Dataset:
    return split_data(load_frame(), test_size=test_size, seed=seed)
