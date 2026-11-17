"""Data validation and cleaning tests (synthetic fixtures)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from air_quality.data_validation import DataValidationError, audit_hourly, validate_schema
from air_quality.download import local_dates_to_utc_unix, plan_chunks
from air_quality.feature_engineering import daily_targets
from air_quality.preprocessing import clean_hourly

TZ = "Asia/Almaty"


def test_duplicate_timestamps_removed(small_raw):
    raw = pd.concat([small_raw, small_raw.iloc[:5]], ignore_index=True)
    conflict = small_raw.iloc[[10]].copy()
    conflict["pm2_5"] = 999.0
    raw = pd.concat([raw, conflict], ignore_index=True)
    clean, rep = clean_hourly(raw, TZ)
    step = [s for s in rep["steps"] if s["step"] == "remove_duplicates"][0]
    assert step["exact_duplicates_removed"] == 5
    assert step["conflicting_removed_keep_first"] == 1
    assert clean["timestamp_utc"].is_unique
    assert clean.loc[clean["dt_unix"] == small_raw.iloc[10]["dt_unix"], "pm2_5"].iloc[0] != 999.0


def test_missing_intervals_kept_as_nan(small_raw):
    raw = small_raw.drop(index=range(100, 110))
    clean, rep = clean_hourly(raw, TZ)
    assert len(clean) == len(small_raw)
    assert (~clean["is_observed"]).sum() == 10
    assert clean.loc[~clean["is_observed"], "pm2_5"].isna().all()
    assert rep["gaps"]["max_gap_hours"] == 10


def test_utc_local_conversion():
    # 2025-06-10 12:00 UTC is 17:00 in Asia/Almaty (UTC+5 since 2024-03-01)
    raw = pd.DataFrame({"dt_unix": [int(pd.Timestamp("2025-06-10 12:00", tz="UTC").timestamp())], "aqi": [1],
                        "co": [1], "no": [1], "no2": [1], "o3": [1], "so2": [1], "pm2_5": [5.0], "pm10": [6], "nh3": [1]})
    clean, _ = clean_hourly(raw, TZ)
    assert clean["hour_local"].iloc[0] == 17
    assert clean["date_local"].iloc[0] == "2025-06-10"
    s, e = local_dates_to_utc_unix("2025-06-10", "2025-06-10", TZ)
    assert pd.Timestamp(s, unit="s", tz="UTC") == pd.Timestamp("2025-06-09 19:00", tz="UTC")
    assert (e - s) == 24 * 3600


def test_negative_values_invalidated(small_raw):
    raw = small_raw.copy()
    raw.loc[raw.index[3], "pm2_5"] = -4.0
    raw.loc[raw.index[4], "aqi"] = 9
    clean, rep = clean_hourly(raw, TZ)
    step = [s for s in rep["steps"] if s["step"] == "invalidate_impossible_values"][0]
    assert step["set_to_nan"]["pm2_5"] == 1 and step["set_to_nan"]["aqi"] == 1
    assert (clean["pm2_5"].dropna() >= 0).all()


def test_extremes_flagged_not_removed(small_raw):
    raw = small_raw.copy()
    raw.loc[raw.index[50], "pm2_5"] = 900.0
    clean, rep = clean_hourly(raw, TZ)
    assert clean["pm2_5"].max() == 900.0
    assert rep["extremes"]["n_flagged_hours"] >= 1


def test_incomplete_days_excluded_from_targets(small_raw):
    raw = small_raw.copy()
    clean, _ = clean_hourly(raw, TZ)
    day = clean["date_local"].unique()[5]
    idx = clean.index[clean["date_local"] == day][:6]  # remove 6 of 24 hours -> 18 < 20
    clean.loc[idx, "pm2_5"] = np.nan
    t = daily_targets(clean, 20).set_index("target_date")
    assert not t.loc[day, "target_is_valid"]
    assert np.isnan(t.loc[day, "target_pm25"])
    assert t.loc[day, "target_valid_hours"] == 18


def test_sorting_and_future_removed(small_raw):
    raw = small_raw.sample(frac=1, random_state=3)
    fut = small_raw.iloc[[0]].copy()
    fut["dt_unix"] = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=3)).timestamp())
    clean, rep = clean_hourly(pd.concat([raw, fut]), TZ)
    assert clean["timestamp_utc"].is_monotonic_increasing
    assert [s for s in rep["steps"] if s["step"] == "drop_future_timestamps"][0]["removed"] == 1


def test_missing_target_values_not_imputed(small_raw):
    clean, _ = clean_hourly(small_raw, TZ)
    day = clean["date_local"].unique()[3]
    clean.loc[clean["date_local"] == day, "pm2_5"] = np.nan
    t = daily_targets(clean, 20).set_index("target_date")
    assert np.isnan(t.loc[day, "target_pm25"])


def test_schema_validation():
    with pytest.raises(DataValidationError):
        validate_schema(pd.DataFrame({"dt_unix": [1]}))
    df = pd.DataFrame({"dt_unix": ["1", "x"], "aqi": [1, 2], "co": ["1.5", "bad"], "no": [1, 1], "no2": [1, 1],
                       "o3": [1, 1], "so2": [1, 1], "pm2_5": [1, 1], "pm10": [1, 1], "nh3": [1, 1]})
    out, bad = validate_schema(df)
    assert bad["co"] == 1 and bad["dt_unix"] == 1


def test_audit_counts(small_raw):
    a = audit_hourly(pd.concat([small_raw, small_raw.iloc[:3]]), TZ)
    assert a["n_duplicated_timestamps"] == 3
    assert a["coverage_pct"] == 100.0


def test_chunk_plan_has_no_overlap():
    ch = plan_chunks(0, 20 * 86400 + 7200, 7)
    assert ch[0] == (0, 7 * 86400)
    assert all(a[1] == b[0] for a, b in zip(ch, ch[1:]))
    assert ch[-1][1] == 20 * 86400 + 7200
