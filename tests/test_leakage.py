"""Leakage tests: no feature may depend on information after the issue time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from air_quality.feature_engineering import (POLLUTION_FEATURES as ALL_FEATURES, FEATURE_SETS, assign_split, build_dataset,
                                             compute_features, issue_time_for_date)
from air_quality.feature_selection import TopKSelector
from air_quality.models import build_pipeline
from air_quality.preprocessing import clean_hourly
from air_quality.training import TARGET

TZ = "Asia/Almaty"
FC = {"issue_hour_local": 18, "rolling_min_coverage": 0.75, "last_value_max_age_hours": 3}


@pytest.fixture
def grid(small_raw):
    g, _ = clean_hourly(small_raw, TZ)
    return g


def _features(grid, date="2025-01-10"):
    issue = issue_time_for_date(date, TZ, 18)
    return compute_features(grid, [issue], FC).iloc[0], issue


def test_future_observations_do_not_change_features(grid):
    base, issue = _features(grid)
    g2 = grid.copy()
    after = g2["timestamp_utc"] >= issue.tz_convert("UTC") - pd.Timedelta(hours=0)
    g2.loc[after, ["pm2_5", "pm10", "no2", "co", "o3", "so2", "nh3", "no", "aqi"]] = 1e6
    pert, _ = _features(g2)
    for f in ALL_FEATURES:
        assert (np.isnan(base[f]) and np.isnan(pert[f])) or base[f] == pytest.approx(pert[f]), f


def test_record_labelled_at_issue_hour_is_not_used(grid):
    """The record labelled 18:00 covers 18:00-19:00 and is unavailable at 18:00."""
    base, issue = _features(grid)
    g2 = grid.copy()
    g2.loc[g2["timestamp_utc"] == issue.tz_convert("UTC"), "pm2_5"] = 5000.0
    pert, _ = _features(g2)
    assert base["pm25_last"] == pert["pm25_last"]
    g3 = grid.copy()
    g3.loc[g3["timestamp_utc"] == issue.tz_convert("UTC") - pd.Timedelta(hours=1), "pm2_5"] = 5000.0
    pert3, _ = _features(g3)
    assert pert3["pm25_last"] == 5000.0  # the 17:00 record IS the newest usable one


def test_target_day_values_not_in_features(grid):
    base, issue = _features(grid)
    tday = (issue.tz_localize(None).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    g2 = grid.copy()
    g2.loc[g2["date_local"] == tday, "pm2_5"] = -1.0
    pert, _ = _features(g2)
    for f in ALL_FEATURES:
        assert (np.isnan(base[f]) and np.isnan(pert[f])) or base[f] == pytest.approx(pert[f]), f


def test_rolling_window_ends_at_cutoff(grid):
    row, issue = _features(grid)
    cut = issue.tz_convert("UTC")
    w = grid[(grid["timestamp_utc"] < cut) & (grid["timestamp_utc"] >= cut - pd.Timedelta(hours=24))]
    assert len(w) == 24
    assert row["pm25_mean_24h"] == pytest.approx(w["pm2_5"].mean())
    w6 = grid[(grid["timestamp_utc"] < cut) & (grid["timestamp_utc"] >= cut - pd.Timedelta(hours=6))]
    assert row["pm25_mean_6h"] == pytest.approx(w6["pm2_5"].mean())
    prev = grid[grid["date_local"] == (issue.tz_localize(None) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    assert row["pm25_prev_day_mean"] == pytest.approx(prev["pm2_5"].mean())


def test_target_is_next_local_day_mean(tmp_project, small_raw):
    g, _ = clean_hourly(small_raw, TZ)
    ds = build_dataset(g, tmp_project)
    r = ds[ds["issue_date"] == "2025-01-10"].iloc[0]
    assert r["target_date"] == "2025-01-11"
    assert r[TARGET] == pytest.approx(g[g["date_local"] == "2025-01-11"]["pm2_5"].mean())
    assert r["issue_time_local"].startswith("2025-01-10T18:00")


def test_chronological_split_by_target_date():
    s = assign_split(pd.Series(["2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2026-01-01"]),
                     {"train": ["2021-01-01", "2023-12-31"], "validation": ["2024-01-01", "2024-12-31"],
                      "test": ["2025-01-01", "2025-12-31"]})
    assert list(s) == ["train", "validation", "validation", "test", "unused"]


def test_scaler_and_imputer_fitted_on_training_only():
    rng = np.random.default_rng(0)
    feats = FEATURE_SETS["basic"]
    Xtr = pd.DataFrame(rng.normal(0, 1, (200, len(feats))), columns=feats)
    Xte = pd.DataFrame(rng.normal(100, 1, (50, len(feats))), columns=feats)
    Xtr.iloc[0, 0] = np.nan
    y = rng.normal(10, 1, 200)
    p = build_pipeline("ridge").fit(Xtr, y)
    assert np.allclose(p.named_steps["scale"].mean_, p.named_steps["impute"].transform(Xtr).mean(axis=0))
    assert np.allclose(p.named_steps["impute"].statistics_, Xtr.median().values)
    p.predict(Xte)  # transform only; statistics unchanged
    assert np.allclose(p.named_steps["impute"].statistics_, Xtr.median().values)


def test_feature_selection_ignores_test_targets():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 10))
    y = 3 * X[:, 0] + rng.normal(size=300)
    tr = slice(0, 200)
    s1 = TopKSelector(3).fit(X[tr], y[tr]).get_support()
    y_perturbed = y.copy()
    y_perturbed[200:] = rng.normal(size=100) * 1000  # change "test" targets only
    s2 = TopKSelector(3).fit(X[tr], y_perturbed[tr]).get_support()
    assert (s1 == s2).all() and s1[0]


def test_dataset_rows_use_only_past(tmp_project, small_raw):
    """Every row's newest record label is strictly before its issue time."""
    g, _ = clean_hourly(small_raw, TZ)
    ds = build_dataset(g, tmp_project).dropna(subset=["meta_newest_record_utc"])
    newest = pd.to_datetime(ds["meta_newest_record_utc"], utc=True)
    issue = pd.to_datetime(ds["issue_time_utc"], utc=True)
    assert (newest + pd.Timedelta(hours=1) <= issue).all()
