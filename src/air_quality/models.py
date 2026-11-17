"""The four trainable regression models, the persistence baseline, and their
hyperparameter search spaces.

Each model is wrapped in a scikit-learn ``Pipeline`` so that all fitted
preprocessing (median imputation, optional feature selection, scaling) is
learned from the training fold only:

    MODEL 1  Ridge Regression        imputer -> [selector] -> StandardScaler -> Ridge
    MODEL 2  Random Forest Regressor imputer -> [selector] -> RandomForestRegressor
    MODEL 3  XGBoost Regressor       imputer -> [selector] -> XGBRegressor
    MODEL 4  Support Vector Reg.     imputer -> [selector] -> StandardScaler -> SVR(RBF)
    MODEL 5  Ensemble                equal-weight average of tuned RF + XGBoost + SVR
    BASELINE Persistence             y_hat = trailing 24-h mean PM2.5 at issue time
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .feature_selection import make_selector

MODEL_NAMES = ["ridge", "random_forest", "xgboost", "svr"]
DISPLAY_NAMES = {"ridge": "Ridge Regression", "random_forest": "Random Forest",
                 "xgboost": "XGBoost", "svr": "SVR (RBF)", "ensemble": "Ensemble (RF+XGB+SVR)",
                 "persistence": "Persistence (24-h mean)"}
ENSEMBLE_MEMBERS = ["random_forest", "xgboost", "svr"]
NEEDS_SCALING = {"ridge": True, "random_forest": False, "xgboost": False, "svr": True}


def _xgb_regressor(seed: int, n_jobs: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("xgboost is not installed. Run: pip install -r requirements.txt") from exc
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_estimators=300,
                        learning_rate=0.1, max_depth=6, random_state=seed, n_jobs=n_jobs, verbosity=0)


def make_estimator(name: str, seed: int = 42, n_jobs: int = -1):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=n_jobs)
    if name == "xgboost":
        return _xgb_regressor(seed, n_jobs)
    if name == "svr":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, select: bool = False, seed: int = 42, n_jobs: int = -1) -> Pipeline:
    """Pipeline with *library-default-like* hyperparameters (used for E4 'default')."""
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if select:
        steps.append(("select", make_selector()))
    if NEEDS_SCALING[name]:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_estimator(name, seed, n_jobs)))
    return Pipeline(steps)


def param_space(name: str, select: bool = False) -> dict[str, Any]:
    """Bounded RandomizedSearchCV distributions (student-laptop budget)."""
    if name == "ridge":
        sp: dict[str, Any] = {"model__alpha": loguniform(1e-3, 1e3)}
    elif name == "random_forest":
        sp = {"model__n_estimators": [200, 300, 500],
              "model__max_depth": [None, 6, 10, 16],
              "model__min_samples_split": [2, 5, 10],
              "model__min_samples_leaf": [1, 2, 4, 8],
              "model__max_features": ["sqrt", 0.33, 0.5, 1.0]}
    elif name == "xgboost":
        sp = {"model__n_estimators": [150, 300, 500],
              "model__max_depth": [2, 3, 4, 6],
              "model__learning_rate": loguniform(0.01, 0.2),
              "model__subsample": [0.6, 0.8, 1.0],
              "model__colsample_bytree": [0.6, 0.8, 1.0],
              "model__reg_lambda": loguniform(0.1, 10.0),
              "model__min_child_weight": [1, 3, 5, 10]}
    elif name == "svr":
        sp = {"model__C": loguniform(0.1, 300.0),
              "model__epsilon": [0.1, 0.5, 1.0, 2.0],
              "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1]}
    else:
        raise ValueError(name)
    if select:
        sp["select__topk__k"] = [8, 12, 16, 20, 25, "all"]
        sp["select__redundancy__threshold"] = [0.9, 0.95, 0.99]
    return sp


def make_ensemble(estimators: dict[str, Any]):
    """MODEL 5: equal-weight average of already-tuned pipelines (VotingRegressor).

    Motivated by Haseeb et al. (2026), where a weighted RF/XGBoost/CatBoost
    ensemble gave the best independent-year accuracy. Equal weights avoid
    fitting weights on validation or test data.
    """
    from sklearn.base import clone
    from sklearn.ensemble import VotingRegressor
    return VotingRegressor([(k, clone(v)) for k, v in estimators.items()])


class PersistenceForecaster(RegressorMixin, BaseEstimator):
    """Non-ML baseline: tomorrow's mean = trailing 24-h mean available at 18:00."""

    def __init__(self, feature: str = "pm25_mean_24h"):
        self.feature = feature

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X[self.feature], dtype=float)
