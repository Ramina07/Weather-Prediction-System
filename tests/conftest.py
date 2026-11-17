"""Shared pytest fixtures.

IMPORTANT: every dataset generated here is SYNTHETIC. It exists only to test
code paths (parsing, cleaning, leakage, model fitting, CLI). It is never
written to data/ and must never be presented as real observations or as
experimental results.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality.config import load_config  # noqa: E402

TZ = "Asia/Almaty"


def synthetic_raw(start="2020-12-20", end="2026-01-02", seed=0, gaps=True, dups=True) -> pd.DataFrame:
    """SYNTHETIC hourly records in the raw CSV schema (dt_unix + pollutants)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    n = len(ts)
    doy = ts.dayofyear.to_numpy()
    hour = ts.tz_convert(TZ).hour.to_numpy()
    seasonal = 12 + 10 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = 3 * np.sin(2 * np.pi * (hour - 20) / 24)
    ar = np.zeros(n)
    eps = rng.normal(0, 2.0, n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + eps[i]
    pm25 = np.clip(seasonal + diurnal + ar, 0.5, None)
    df = pd.DataFrame({
        "dt_unix": [int(t.timestamp()) for t in ts],
        "aqi": np.clip(np.ceil(pm25 / 15), 1, 5),
        "co": 300 + 8 * pm25 + rng.normal(0, 20, n),
        "no": np.abs(rng.normal(1, 0.5, n)),
        "no2": 10 + 0.5 * pm25 + rng.normal(0, 2, n),
        "o3": np.abs(60 - 0.8 * pm25 + rng.normal(0, 5, n)),
        "so2": np.abs(5 + 0.2 * pm25 + rng.normal(0, 1, n)),
        "pm2_5": pm25,
        "pm10": pm25 * 1.3 + np.abs(rng.normal(0, 2, n)),
        "nh3": np.abs(rng.normal(2, 0.5, n)),
    })
    if gaps:
        drop = np.zeros(n, bool)
        for s in rng.choice(n - 50, 25, replace=False):
            drop[s:s + rng.integers(1, 30)] = True
        df = df[~drop]
    if dups:
        df = pd.concat([df, df.sample(10, random_state=seed)]).reset_index(drop=True)
    return df


def synthetic_weather(start="2020-12-15", end="2026-01-05", seed=0) -> pd.DataFrame:
    """SYNTHETIC hourly weather in the Open-Meteo-derived schema."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    doy = ts.dayofyear.to_numpy()
    n = len(ts)
    return pd.DataFrame({"timestamp_utc": ts,
                         "temp": -5 + 20 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 3, n),
                         "rh": np.clip(70 + rng.normal(0, 10, n), 5, 100),
                         "pressure": 980 + rng.normal(0, 5, n),
                         "precip": np.clip(rng.normal(0, 0.3, n), 0, None),
                         "cloud": np.clip(rng.normal(50, 30, n), 0, 100),
                         "wind": np.abs(rng.normal(4, 2, n)),
                         "wind_dir": rng.uniform(0, 360, n)})


@pytest.fixture
def tmp_project(tmp_path: Path):
    """A temporary project root with a copy of config.yaml (fast settings)."""
    data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data["training"]["n_iter"] = 2
    data["training"]["n_iter_quick"] = 2
    data["training"]["cv_splits"] = 3
    data["training"]["n_jobs"] = 1
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def raw_synth():
    return synthetic_raw()


@pytest.fixture
def small_raw():
    return synthetic_raw("2024-12-01", "2025-01-20", seed=1, gaps=False, dups=False)


def have_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False
