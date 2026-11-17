"""Leakage-free target and feature construction at a fixed issue time.

Forecast protocol (see docs/research_methodology.md):

* Issue time: ``issue_hour_local`` (18:00) Asia/Almaty on local date d.
* Target: mean PM2.5 over all valid hourly records on local date d+1
  (00:00-23:00 hour labels), valid only if at least ``target_min_valid_hours``
  hours are present.
* Timestamp convention: a record labelled ``hh:00`` covers ``[hh:00, hh+1:00)``
  and becomes available at ``hh+1:00``. At an 18:00 issue the newest usable
  record is labelled 17:00. Every feature uses only records with
  ``label + 1 h <= issue time``.

The single function :func:`compute_features` is used for BOTH historical
training samples and live inference, guaranteeing identical feature logic.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config, get_logger
from .weather import WEATHER_FEATURES, WEATHER_PAST_FEATURES

FEATURE_VERSION = "1.0"
OTHER_POLLUTANTS = ["pm10", "no2", "co", "o3", "so2", "nh3", "no"]
LAGGED_OTHERS = ["pm10", "no2", "co", "o3"]
ROLL_WINDOWS = [6, 12, 24, 48, 72]

CALENDAR_FEATURES = ["target_dow", "target_month", "target_season", "target_is_weekend",
                     "target_month_sin", "target_month_cos", "target_doy_sin", "target_doy_cos",
                     "target_is_heating_season"]
BASIC_HIST = ["pm25_last"]
LAG_FEATURES = (["pm25_lag_24h", "pm25_lag_48h", "pm25_lag_72h"]
                + [f"{p}_last" for p in OTHER_POLLUTANTS]
                + [f"{p}_lag_24h" for p in LAGGED_OTHERS])
ROLL_FEATURES = ([f"pm25_mean_{w}h" for w in ROLL_WINDOWS]
                 + ["pm25_std_24h", "pm25_min_24h", "pm25_max_24h", "pm25_trend_24h", "pm25_diff_24h",
                    "pm25_prev_day_mean", "pm_ratio_24h"]
                 + [f"{p}_mean_24h" for p in OTHER_POLLUTANTS])
AQI_FEATURES = ["aqi_last", "aqi_mean_24h"]

FEATURE_SETS: dict[str, list[str]] = {
    "basic": CALENDAR_FEATURES + BASIC_HIST,
    "lag": CALENDAR_FEATURES + BASIC_HIST + LAG_FEATURES,
    "full": CALENDAR_FEATURES + BASIC_HIST + LAG_FEATURES + ROLL_FEATURES,
    "full_aqi": CALENDAR_FEATURES + BASIC_HIST + LAG_FEATURES + ROLL_FEATURES + AQI_FEATURES,
    # weather-augmented sets (see weather.py for the perfect-prognosis note)
    "full_weather": CALENDAR_FEATURES + BASIC_HIST + LAG_FEATURES + ROLL_FEATURES + WEATHER_FEATURES,
    "full_weather_past": CALENDAR_FEATURES + BASIC_HIST + LAG_FEATURES + ROLL_FEATURES + WEATHER_PAST_FEATURES,
}
# "full_selected" uses the "full" columns followed by an in-pipeline selector.
CONFIG_BASE_SET = {"basic": "basic", "lag": "lag", "full": "full", "full_selected": "full",
                   "full_aqi": "full_aqi", "full_weather": "full_weather", "full_weather_past": "full_weather_past"}
POLLUTION_FEATURES = FEATURE_SETS["full_aqi"]
ALL_FEATURES = FEATURE_SETS["full_aqi"] + WEATHER_FEATURES
WEATHER_CONFIGS = {"full_weather", "full_weather_past"}
PERSISTENCE_FEATURE = "pm25_mean_24h"

META_COLUMNS = ["issue_date", "issue_time_local", "issue_time_utc", "target_date", "target_pm25",
                "target_valid_hours", "target_is_valid", "meta_pm25_coverage_72h",
                "meta_last_obs_age_hours", "meta_newest_record_utc", "eligible"]


def features_for_config(config_name: str) -> list[str]:
    try:
        return FEATURE_SETS[CONFIG_BASE_SET[config_name]]
    except KeyError:
        raise ValueError(f"Unknown feature configuration: {config_name}") from None


# --------------------------------------------------------------------------- calendar
def season_of_month(m: int) -> int:
    """0 = winter (DJF), 1 = spring (MAM), 2 = summer (JJA), 3 = autumn (SON)."""
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(m, 3)


def is_heating_season(date: pd.Timestamp, start_mmdd: str = "10-15", end_mmdd: str = "04-15") -> bool:
    md = date.strftime("%m-%d")
    return md >= start_mmdd or md <= end_mmdd


def calendar_features(target_date: pd.Timestamp, heating: dict[str, str] | None = None) -> dict[str, float]:
    heating = heating or {"start": "10-15", "end": "04-15"}
    m, doy = target_date.month, target_date.dayofyear
    return {
        "target_dow": float(target_date.dayofweek),
        "target_month": float(m),
        "target_season": float(season_of_month(m)),
        "target_is_weekend": float(target_date.dayofweek >= 5),
        "target_month_sin": float(np.sin(2 * np.pi * (m - 1) / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * (m - 1) / 12)),
        "target_doy_sin": float(np.sin(2 * np.pi * (doy - 1) / 365.25)),
        "target_doy_cos": float(np.cos(2 * np.pi * (doy - 1) / 365.25)),
        "target_is_heating_season": float(is_heating_season(target_date, heating["start"], heating["end"])),
    }


# --------------------------------------------------------------------------- helpers
def _take(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather ``arr[idx]``; out-of-range indices yield NaN (unknown history)."""
    out = np.full(idx.shape, np.nan)
    ok = (idx >= 0) & (idx < len(arr))
    out[ok] = arr[idx[ok]]
    return out


def _cov_ok(w: np.ndarray, min_cov: float) -> bool:
    return w.size > 0 and np.isfinite(w).sum() / w.size >= min_cov


def _mean(w, c):
    return float(np.nanmean(w)) if _cov_ok(w, c) else np.nan


def _std(w, c):
    return float(np.nanstd(w, ddof=1)) if _cov_ok(w, c) and np.isfinite(w).sum() > 1 else np.nan


def _min(w, c):
    return float(np.nanmin(w)) if _cov_ok(w, c) else np.nan


def _max(w, c):
    return float(np.nanmax(w)) if _cov_ok(w, c) else np.nan


def _slope(w, c):
    """OLS slope (ug/m3 per hour) over the valid points of window ``w``."""
    if not _cov_ok(w, c):
        return np.nan
    x = np.arange(w.size, dtype=float)
    m = np.isfinite(w)
    if m.sum() < 3:
        return np.nan
    xm, ym = x[m] - x[m].mean(), w[m] - w[m].mean()
    den = (xm ** 2).sum()
    return float((xm * ym).sum() / den) if den > 0 else np.nan


def _last_valid(w: np.ndarray, max_age: int) -> tuple[float, float]:
    """Most recent finite value within the last ``max_age`` hours; (value, age)."""
    for age in range(min(max_age, w.size)):
        v = w[w.size - 1 - age]
        if np.isfinite(v):
            return float(v), float(age)
    return np.nan, np.nan


# --------------------------------------------------------------------------- core
def issue_time_for_date(issue_date: Any, tz: str, issue_hour: int) -> pd.Timestamp:
    d = pd.Timestamp(issue_date)
    d = d.tz_localize(None) if d.tzinfo else d
    return pd.Timestamp(d.normalize() + pd.Timedelta(hours=issue_hour)).tz_localize(tz)


def compute_features(grid: pd.DataFrame, issue_times_local: Iterable[pd.Timestamp], fc: dict[str, Any],
                     heating: dict[str, str] | None = None) -> pd.DataFrame:
    """Compute all features for each issue time from a complete hourly grid.

    ``grid`` must contain a tz-aware ``timestamp_utc`` column on a regular
    hourly grid (as produced by :func:`preprocessing.clean_hourly`) and the
    pollutant columns. Records with ``label + 1h > issue`` are never read.
    """
    g = grid.sort_values("timestamp_utc").reset_index(drop=True)
    ts = g["timestamp_utc"]
    t0 = ts.iloc[0]
    hour = pd.Timedelta(hours=1)
    n_rows = len(g)
    arrays = {c: g[c].to_numpy(dtype=float) for c in ["pm2_5", "aqi", *OTHER_POLLUTANTS] if c in g.columns}
    if "is_observed" in g.columns:  # grid placeholders are NaN already; keep explicit
        obs = g["is_observed"].to_numpy(dtype=bool)
        for k in arrays:
            arrays[k] = np.where(obs, arrays[k], np.nan)
    min_cov = fc.get("rolling_min_coverage", 0.75)
    max_age = int(fc.get("last_value_max_age_hours", 3))

    rows = []
    for issue_local in issue_times_local:
        issue_utc = issue_local.tz_convert("UTC")
        # index of the newest usable label = issue - 1h
        last_idx = int((issue_utc - hour - t0) // hour)
        newest_avail = min(last_idx, n_rows - 1)
        newest_record = ts.iloc[newest_avail] if newest_avail >= 0 else pd.NaT

        def win(name: str, n: int) -> np.ndarray:
            return _take(arrays[name], np.arange(last_idx - n + 1, last_idx + 1))

        def at(name: str, lag: int) -> float:
            return float(_take(arrays[name], np.array([last_idx - lag]))[0])

        pm72 = win("pm2_5", 72)
        pm24 = pm72[-24:]
        r: dict[str, Any] = {}
        r["pm25_last"], age = _last_valid(pm72, max_age)
        for L in (24, 48, 72):
            r[f"pm25_lag_{L}h"] = at("pm2_5", L)
        for w in ROLL_WINDOWS:
            r[f"pm25_mean_{w}h"] = _mean(pm72[-w:], min_cov)
        r["pm25_std_24h"] = _std(pm24, min_cov)
        r["pm25_min_24h"] = _min(pm24, min_cov)
        r["pm25_max_24h"] = _max(pm24, min_cov)
        r["pm25_trend_24h"] = _slope(pm24, min_cov)
        prev24 = _mean(pm72[-48:-24], min_cov)
        r["pm25_diff_24h"] = r["pm25_mean_24h"] - prev24 if np.isfinite(prev24) else np.nan
        # previous LOCAL calendar day (d-1), fully observed before the issue time
        day_d = pd.Timestamp(issue_local.date())
        d_start = day_d.tz_localize(issue_local.tz)
        p_start = (day_d - pd.Timedelta(days=1)).tz_localize(issue_local.tz)
        i0 = int((p_start.tz_convert("UTC") - t0) // hour)
        i1 = int((d_start.tz_convert("UTC") - t0) // hour)
        assert i1 - 1 <= last_idx, "previous-day window must end before the issue time"
        r["pm25_prev_day_mean"] = _mean(_take(arrays["pm2_5"], np.arange(i0, i1)), min_cov)
        for p in OTHER_POLLUTANTS:
            if p not in arrays:
                r[f"{p}_last"] = r[f"{p}_mean_24h"] = np.nan
                continue
            w24 = win(p, 24)
            r[f"{p}_last"], _ = _last_valid(w24, max_age)
            r[f"{p}_mean_24h"] = _mean(w24, min_cov)
        for p in LAGGED_OTHERS:
            r[f"{p}_lag_24h"] = at(p, 24) if p in arrays else np.nan
        pm10m = r["pm10_mean_24h"]
        r["pm_ratio_24h"] = (r["pm25_mean_24h"] / pm10m) if np.isfinite(pm10m) and pm10m > 0 else np.nan
        if "aqi" in arrays:
            a24 = win("aqi", 24)
            r["aqi_last"], _ = _last_valid(a24, max_age)
            r["aqi_mean_24h"] = _mean(a24, min_cov)
        else:
            r["aqi_last"] = r["aqi_mean_24h"] = np.nan
        target_date = (issue_local.tz_localize(None).normalize() + pd.Timedelta(days=1))
        r.update(calendar_features(target_date, heating))
        r["issue_date"] = issue_local.strftime("%Y-%m-%d")
        r["issue_time_local"] = issue_local.isoformat()
        r["issue_time_utc"] = issue_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        r["target_date"] = target_date.strftime("%Y-%m-%d")
        r["meta_pm25_coverage_72h"] = float(np.isfinite(pm72).mean())
        r["meta_last_obs_age_hours"] = age
        r["meta_newest_record_utc"] = newest_record.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(newest_record) else None
        rows.append(r)
    return pd.DataFrame(rows)


def daily_targets(grid: pd.DataFrame, min_valid_hours: int) -> pd.DataFrame:
    """Daily mean PM2.5 per local date from observed hours only."""
    obs = grid[grid["is_observed"]] if "is_observed" in grid.columns else grid
    d = obs.groupby("date_local")["pm2_5"].agg(target_pm25="mean", target_valid_hours="count").reset_index()
    d["target_is_valid"] = d["target_valid_hours"] >= min_valid_hours
    d.loc[~d["target_is_valid"], "target_pm25"] = np.nan  # never use insufficient days
    return d.rename(columns={"date_local": "target_date"})


def build_dataset(grid: pd.DataFrame, cfg: Config, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per issue date d: features at d 18:00 and target for d+1."""
    fc = {**cfg["forecast"], "_tz": cfg.tz}
    dates = pd.to_datetime(sorted(grid["date_local"].unique()))
    issue_dates = dates[:-1]  # last date has no next-day target
    issues = [issue_time_for_date(d, cfg.tz, fc["issue_hour_local"]) for d in issue_dates]
    feats = compute_features(grid, issues, fc, cfg["evaluation"]["heating_season"])
    tgt = daily_targets(grid, fc["target_min_valid_hours"])
    ds = feats.merge(tgt, on="target_date", how="left")
    if weather is not None and len(weather):
        from .weather import compute_weather_features
        wf = compute_weather_features(weather, issues)
        ds = ds.merge(wf, on="issue_date", how="left")
    for c in WEATHER_FEATURES:
        if c not in ds.columns:
            ds[c] = np.nan
    ds["target_valid_hours"] = ds["target_valid_hours"].fillna(0).astype(int)
    ds["target_is_valid"] = ds["target_is_valid"].fillna(False).astype(bool)
    ds["eligible"] = ds["target_is_valid"] & ds["pm25_last"].notna() & ds[PERSISTENCE_FEATURE].notna()
    cols = META_COLUMNS + ALL_FEATURES
    return ds[cols]


def assign_split(target_dates: pd.Series, splits: dict[str, list[str]]) -> pd.Series:
    td = pd.to_datetime(target_dates)
    out = pd.Series("unused", index=target_dates.index, dtype=object)
    for name in ("train", "validation", "test"):
        a, b = pd.Timestamp(splits[name][0]), pd.Timestamp(splits[name][1])
        out[(td >= a) & (td <= b)] = name
    return out


def run_features(cfg: Config, operational: bool = False) -> pd.DataFrame:
    from .preprocessing import load_clean
    log = get_logger()
    src = cfg.path("clean_operational_csv" if operational else "clean_hourly_csv")
    if not src.exists():
        raise FileNotFoundError(f"Clean hourly data not found: {src}. Run the preprocess command first.")
    grid = load_clean(src, cfg.tz)
    weather = None
    if cfg.path("weather_csv").exists():
        from .weather import load_weather
        weather = load_weather(cfg.path("weather_csv"))
    else:
        log.warning("No weather file (%s): weather features will be empty. Run the `weather` command.",
                    cfg.path("weather_csv"))
    ds = build_dataset(grid, cfg, weather)
    ds["split"] = assign_split(ds["target_date"], cfg["splits"]) if not operational else "operational"
    out = cfg.path("features_operational_csv" if operational else "features_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out, index=False)
    log.info("Feature dataset: %d issue dates, %d eligible -> %s", len(ds), int(ds["eligible"].sum()), out)
    if not operational:
        log.info("Eligible samples per split: %s",
                 ds[ds["eligible"]].groupby("split").size().to_dict())
    return ds
