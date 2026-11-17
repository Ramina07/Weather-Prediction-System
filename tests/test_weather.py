"""Weather parsing and leakage tests (SYNTHETIC data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from air_quality.feature_engineering import issue_time_for_date
from air_quality.weather import (WEATHER_PAST_FEATURES, WeatherError, compute_weather_features,
                                 parse_open_meteo)

from .conftest import synthetic_weather

TZ = "Asia/Almaty"


def test_parse_open_meteo():
    pl = {"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"], "temperature_2m": [-10.5, -11.0],
                     "relative_humidity_2m": [80, 82], "surface_pressure": [990, 991], "precipitation": [0, 0.1],
                     "cloud_cover": [100, 90], "wind_speed_10m": [1.2, 3.4], "wind_direction_10m": [200, 210]}}
    df = parse_open_meteo(pl)
    assert list(df["temp"]) == [-10.5, -11.0]
    assert str(df["timestamp_utc"].dt.tz) == "UTC"


def test_parse_error():
    with pytest.raises(WeatherError):
        parse_open_meteo({"error": True, "reason": "bad request"})


def test_past_weather_ignores_future():
    w = synthetic_weather("2025-01-01", "2025-01-20")
    issue = issue_time_for_date("2025-01-10", TZ, 18)
    base = compute_weather_features(w, [issue]).iloc[0]
    w2 = w.copy()
    w2.loc[w2["timestamp_utc"] >= issue.tz_convert("UTC"), ["temp", "rh", "wind", "pressure", "cloud", "precip"]] = 999
    pert = compute_weather_features(w2, [issue]).iloc[0]
    for f in WEATHER_PAST_FEATURES:
        assert base[f] == pytest.approx(pert[f]), f
    assert pert["wxf_temp_mean"] == 999  # target-day block DOES read d+1 (weather forecast input)


def test_target_day_window_is_next_local_day():
    w = synthetic_weather("2025-06-01", "2025-06-10")
    issue = issue_time_for_date("2025-06-05", TZ, 18)
    r = compute_weather_features(w, [issue]).iloc[0]
    day = w[(w.timestamp_utc.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d") == "2025-06-06")]
    assert len(day) == 24
    assert r["wxf_temp_mean"] == pytest.approx(day["temp"].mean())
    assert r["wxf_calm_hours"] == (day["wind"] < 2).sum()
