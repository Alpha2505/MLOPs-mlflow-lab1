"""Model factory: every model is an sklearn Pipeline so preprocessing is logged with it."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "random_forest": {"n_estimators": 200, "max_depth": 6},
    "grad_boost": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
}
MODEL_CHOICES = tuple(DEFAULT_PARAMS)


def resolved_params(name: str, params: dict | None = None) -> dict:
    """The hyperparameters actually used (defaults merged with overrides)."""
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_CHOICES}.")
    return {**DEFAULT_PARAMS[name], **(params or {})}


def build_model(name: str, params: dict | None = None, seed: int = 42) -> Pipeline:
    p = resolved_params(name, params)
    if name == "logreg":
        clf = LogisticRegression(C=p["C"], max_iter=5000, random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=seed
        )
    else:
        clf = GradientBoostingClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            random_state=seed,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
