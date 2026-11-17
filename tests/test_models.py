"""Model tests: fitting, numeric output, persistence, reproducibility."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from air_quality.feature_engineering import FEATURE_SETS
from air_quality.models import MODEL_NAMES, PersistenceForecaster, build_pipeline, param_space

from .conftest import have_xgboost

FEATS = FEATURE_SETS["full"]


def _xy(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(10, 3, (n, len(FEATS))), columns=FEATS)
    X.iloc[::17, 3] = np.nan
    y = 0.8 * X["pm25_mean_24h"].to_numpy() + rng.normal(0, 1, n)
    return X, y


def _models():
    return [m for m in MODEL_NAMES if m != "xgboost" or have_xgboost()]


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("select", [False, True])
def test_models_fit_and_predict_numeric(name, select):
    if name == "xgboost" and not have_xgboost():
        pytest.skip("xgboost not installed in this environment")
    X, y = _xy()
    p = build_pipeline(name, select=select, n_jobs=1).fit(X, y)
    pred = p.predict(X.iloc[:10])
    assert pred.shape == (10,) and np.isfinite(pred).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_param_spaces_are_bounded(name):
    sp = param_space(name, select=True)
    assert "select__topk__k" in sp and all(k.startswith(("model__", "select__")) for k in sp)


def test_saved_model_reloads_without_retraining(tmp_path):
    X, y = _xy()
    p = build_pipeline("random_forest", n_jobs=1).fit(X, y)
    joblib.dump({"pipeline": p, "features": FEATS}, tmp_path / "m.joblib")
    b = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(b["pipeline"].predict(X[b["features"]]), p.predict(X))


@pytest.mark.parametrize("name", ["random_forest", "svr", "ridge"] + (["xgboost"] if have_xgboost() else []))
def test_reproducible_with_seed(name):
    X, y = _xy()
    a = build_pipeline(name, seed=7, n_jobs=1).fit(X, y).predict(X)
    b = build_pipeline(name, seed=7, n_jobs=1).fit(X, y).predict(X)
    assert np.allclose(a, b)


def test_persistence_uses_trailing_24h_mean():
    X = pd.DataFrame({"pm25_mean_24h": [5.0, 7.5]})
    assert list(PersistenceForecaster().fit(X).predict(X)) == [5.0, 7.5]


def test_prediction_requires_training_schema():
    X, y = _xy()
    p = build_pipeline("ridge").fit(X, y)
    with pytest.raises(ValueError):
        p.predict(X[FEATS[::-1]])  # reordered columns are rejected by sklearn
