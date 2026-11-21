"""Read-only access to the backend outputs (CSV/JSON files written by the pipeline).

Every loader is cached by Streamlit, so pages stay fast. Nothing here calls the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import paths

try:  # Streamlit caching when running inside the app; identity otherwise (tests, scripts)
    import streamlit as st

    cache_data = st.cache_data(show_spinner=False)
    cache_resource = st.cache_resource(show_spinner=False)
except Exception:  # pragma: no cover
    def cache_data(f):
        return f

    cache_resource = cache_data

# --------------------------------------------------------------------------- vocabulary
MODEL_ORDER = ["ensemble", "svr", "xgboost", "random_forest", "ridge", "persistence"]
MODEL_LABELS = {
    "ensemble": "Ensemble (RF + XGB + SVR)",
    "svr": "SVR (RBF kernel)",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "ridge": "Ridge Regression",
    "persistence": "Persistence baseline",
}
MODEL_SHORT = {"ensemble": "Ensemble", "svr": "SVR", "xgboost": "XGBoost", "random_forest": "Random Forest",
               "ridge": "Ridge", "persistence": "Persistence"}
MODEL_COLORS = {
    "ensemble": "#0F766E",
    "svr": "#7C3AED",
    "xgboost": "#EA580C",
    "random_forest": "#16A34A",
    "ridge": "#2563EB",
    "persistence": "#94A3B8",
    "actual": "#0F172A",
}
MODEL_FAMILY = {
    "ensemble": "Averaging ensemble",
    "svr": "Kernel method",
    "xgboost": "Gradient-boosted trees",
    "random_forest": "Bagged trees",
    "ridge": "Regularised linear model",
    "persistence": "Naive baseline",
}
MODEL_IDEA = {
    "ridge": "Linear combination of all features with an L2 penalty that keeps correlated weights stable.",
    "random_forest": "Average of hundreds of decision trees grown on bootstrap samples; captures thresholds "
                     "such as cold and calm days.",
    "xgboost": "Trees added sequentially, each one correcting the residuals of the previous ones, with "
               "regularisation.",
    "svr": "Smooth non-linear fit in an RBF kernel space; errors smaller than ε are ignored.",
    "ensemble": "Equal-weight average of the tuned Random Forest, XGBoost and SVR; their errors partly cancel.",
    "persistence": "Tomorrow equals the last 24 hours. The reference every model has to beat.",
}
CODE_MAP = {
    "ridge": "models.make_estimator('ridge') -> Pipeline(imputer, scaler, Ridge)",
    "random_forest": "models.make_estimator('random_forest') -> Pipeline(imputer, RandomForestRegressor)",
    "xgboost": "models.make_estimator('xgboost') -> Pipeline(imputer, XGBRegressor)",
    "svr": "models.make_estimator('svr') -> Pipeline(imputer, scaler, SVR(rbf))",
    "ensemble": "models.make_ensemble(...) -> VotingRegressor(RF, XGB, SVR)",
    "persistence": "models.PersistenceForecaster (trailing 24-h mean)",
}
CONFIG_LABELS = {
    "basic": "Basic (calendar + last values)",
    "lag": "Lag (+ 24/48/72 h lags)",
    "full": "Full (+ rolling statistics)",
    "full_selected": "Full, feature-selected",
    "full_weather": "Full + weather",
    "full_aqi": "Full + AQI (ablation)",
    "full_weather_past": "Full + past weather only (ablation)",
    "pm25_mean_24h": "Trailing 24-h mean",
}
CONFIG_ORDER = ["basic", "lag", "full", "full_selected", "full_weather", "full_weather_past", "full_aqi"]

# US EPA 24-hour PM2.5 categories (2024 revision of the NAAQS breakpoints)
AQ_CATEGORIES = [
    (0.0, 9.0, "Good", "#22C55E", "Air quality is satisfactory. A great day for outdoor activities."),
    (9.0, 35.4, "Moderate", "#EAB308", "Acceptable. Unusually sensitive people may want to limit long outdoor exertion."),
    (35.4, 55.4, "Unhealthy for sensitive groups", "#F97316",
     "Children, older adults and people with asthma or heart disease should reduce prolonged outdoor exertion."),
    (55.4, 125.4, "Unhealthy", "#EF4444", "Everyone may feel effects. Keep outdoor activity short; consider a mask."),
    (125.4, 225.4, "Very unhealthy", "#9333EA", "Health alert. Avoid outdoor exertion; keep windows closed."),
    (225.4, 1e9, "Hazardous", "#7F1D1D", "Emergency conditions. Stay indoors and use air purification."),
]
WHO_24H = 15.0


def aq_category(value: float | None) -> dict[str, Any]:
    if value is None or not np.isfinite(value):
        return {"name": "Unknown", "color": "#94A3B8", "advice": "No value available.", "index": -1}
    for i, (lo, hi, name, color, advice) in enumerate(AQ_CATEGORIES):
        if value <= hi:
            return {"name": name, "color": color, "advice": advice, "index": i, "lo": lo, "hi": hi}
    return {"name": "Hazardous", "color": "#7F1D1D", "advice": AQ_CATEGORIES[-1][4], "index": 5}


# Feature vocabulary (group + plain-English description)
FEATURE_GROUPS = {
    "Calendar": "#64748B",
    "PM2.5 history": "#0F766E",
    "Other pollutants": "#B45309",
    "Past weather": "#2563EB",
    "Tomorrow's weather": "#7C3AED",
    "AQI": "#DB2777",
}


def feature_group(f: str) -> str:
    if f.startswith("target_"):
        return "Calendar"
    if f.startswith(("pm25_", "pm_ratio")):
        return "PM2.5 history"
    if f.startswith("wxf_"):
        return "Tomorrow's weather"
    if f.startswith("wx_"):
        return "Past weather"
    if f.startswith("aqi"):
        return "AQI"
    return "Other pollutants"


_FEATURE_TEXT = {
    "target_dow": "Day of week of the target day", "target_month": "Month of the target day",
    "target_season": "Season of the target day", "target_is_weekend": "Target day is a weekend",
    "target_month_sin": "Month (cyclic sine)", "target_month_cos": "Month (cyclic cosine)",
    "target_doy_sin": "Day of year (cyclic sine)", "target_doy_cos": "Day of year (cyclic cosine)",
    "target_is_heating_season": "Heating season (15 Oct - 15 Apr)",
    "pm25_last": "Last hourly PM2.5 (17:00)", "pm25_lag_24h": "PM2.5 24 h earlier",
    "pm25_lag_48h": "PM2.5 48 h earlier", "pm25_lag_72h": "PM2.5 72 h earlier",
    "pm25_mean_6h": "Mean PM2.5, last 6 h", "pm25_mean_12h": "Mean PM2.5, last 12 h",
    "pm25_mean_24h": "Mean PM2.5, last 24 h", "pm25_mean_48h": "Mean PM2.5, last 48 h",
    "pm25_mean_72h": "Mean PM2.5, last 72 h", "pm25_std_24h": "PM2.5 variability, last 24 h",
    "pm25_min_24h": "Min PM2.5, last 24 h", "pm25_max_24h": "Max PM2.5, last 24 h",
    "pm25_trend_24h": "PM2.5 trend (slope), last 24 h", "pm25_diff_24h": "PM2.5 change vs 24 h ago",
    "pm25_prev_day_mean": "Mean PM2.5 of the previous full day", "pm_ratio_24h": "PM2.5 / PM10 ratio, last 24 h",
    "wx_temp_mean_24h": "Temperature, last 24 h", "wx_rh_mean_24h": "Humidity, last 24 h",
    "wx_wind_mean_24h": "Wind speed, last 24 h", "wx_cloud_mean_24h": "Cloud cover, last 24 h",
    "wx_precip_sum_24h": "Precipitation, last 24 h", "wx_pressure_last": "Surface pressure at issue time",
    "wx_pressure_change_24h": "Pressure change, last 24 h",
    "wxf_temp_mean": "Forecast mean temperature tomorrow", "wxf_temp_min": "Forecast min temperature tomorrow",
    "wxf_hdd": "Heating degree-days tomorrow", "wxf_wind_mean": "Forecast wind speed tomorrow",
    "wxf_calm_hours": "Calm hours tomorrow (wind < 2 m/s)", "wxf_rh_mean": "Forecast humidity tomorrow",
    "wxf_precip_sum": "Forecast precipitation tomorrow", "wxf_cloud_mean": "Forecast cloud cover tomorrow",
    "wxf_pressure_mean": "Forecast pressure tomorrow", "wxf_temp_change": "Temperature change today -> tomorrow",
    "wxf_pressure_change": "Pressure change today -> tomorrow", "aqi_last": "Provider AQI (last hour)",
    "aqi_mean_24h": "Provider AQI, last 24 h mean",
}


def feature_text(f: str) -> str:
    if f in _FEATURE_TEXT:
        return _FEATURE_TEXT[f]
    for pol in ("pm10", "no2", "co", "o3", "so2", "nh3", "no"):
        if f.startswith(pol + "_"):
            rest = f[len(pol) + 1:]
            nice = {"last": "last hour", "lag_24h": "24 h earlier", "mean_24h": "mean of last 24 h"}.get(rest, rest)
            return f"{pol.upper().replace('PM10', 'PM10')} ({nice})"
    return f


# --------------------------------------------------------------------------- loaders
def _csv(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@cache_data
def model_comparison() -> pd.DataFrame:
    return _csv(paths.METRICS / "model_comparison.csv")


@cache_data
def all_results() -> pd.DataFrame:
    return _csv(paths.METRICS / "all_results.csv")


@cache_data
def metric_table(name: str) -> pd.DataFrame:
    return _csv(paths.METRICS / name)


@cache_data
def metric_json(name: str) -> dict[str, Any]:
    return _json(paths.METRICS / name)


@cache_data
def model_selection() -> dict[str, Any]:
    return _json(paths.METRICS / "model_selection.json")


@cache_data
def model_card() -> dict[str, Any]:
    return _json(paths.MODELS / "benchmark" / "model_card.json")


@cache_data
def latest_forecast_file() -> dict[str, Any]:
    return _json(paths.PREDICTIONS / "latest_forecast.json")


@cache_data
def test_predictions() -> pd.DataFrame:
    df = _csv(paths.PREDICTIONS / "predictions_test.csv")
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@cache_data
def validation_predictions() -> pd.DataFrame:
    df = _csv(paths.PREDICTIONS / "predictions_validation.csv")
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@cache_data
def selected_test_predictions(feature_config: str = "full_weather") -> pd.DataFrame:
    """Wide table: one row per 2025 target date, one column per model (tuned) + actual + persistence."""
    df = test_predictions()
    if df.empty:
        return df
    sel = df[((df["variant"] == "tuned") & (df["feature_config"] == feature_config)) | (df["model"] == "persistence")]
    wide = sel.pivot_table(index="target_date", columns="model", values="predicted", aggfunc="first")
    actual = sel.groupby("target_date")["actual"].first()
    wide.insert(0, "actual", actual)
    return wide.sort_index()


@cache_data
def hourly_pollution() -> pd.DataFrame:
    df = _csv(paths.DATA / "interim" / "hourly_air_pollution_clean.csv")
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["date_local"] = pd.to_datetime(df["date_local"])
    return df


@cache_data
def hourly_weather() -> pd.DataFrame:
    df = _csv(paths.DATA / "raw" / "weather_openmeteo_hourly.csv")
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@cache_data
def daily_features() -> pd.DataFrame:
    df = _csv(paths.DATA / "processed" / "daily_pm25_features.csv")
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@cache_data
def daily_pollution() -> pd.DataFrame:
    """Daily local-calendar means of all pollutants (valid days only: >= 20 observed hours)."""
    h = hourly_pollution()
    if h.empty:
        return h
    cols = ["pm2_5", "pm10", "no2", "co", "o3", "so2", "nh3", "no", "aqi"]
    g = h.groupby("date_local")
    d = g[cols].mean()
    d["n_hours"] = g["pm2_5"].count()
    d = d[d["n_hours"] >= 20]
    return d


@cache_data
def literature_table() -> pd.DataFrame:
    return _csv(paths.DOCS / "literature_comparison.csv")


@cache_data
def doc_text(name: str) -> str:
    p = paths.DOCS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def figure_files() -> list[Path]:
    if not paths.FIGURES.exists():
        return []
    return sorted(p for p in paths.FIGURES.glob("*.png"))


def headline() -> dict[str, Any]:
    """Key numbers used on the home page."""
    mc = model_comparison()
    sel = model_selection()
    audit = metric_json("dataset_audit.json")
    out: dict[str, Any] = {"best_model": sel.get("best", {}).get("model", "ensemble"),
                           "best_config": sel.get("best", {}).get("feature_config", "full_weather"),
                           "n_records": audit.get("n_records"), "coverage_pct": audit.get("coverage_pct"),
                           "n_samples": sel.get("n_samples", {}), "periods": sel.get("periods", {})}
    if not mc.empty:
        row = mc[mc["model"] == out["best_model"]]
        pers = mc[mc["model"] == "persistence"]
        if len(row):
            r = row.iloc[0]
            out.update(test_mae=float(r["test_mae"]), test_rmse=float(r["test_rmse"]), test_r2=float(r["test_r2"]),
                       val_mae=float(r["validation_mae"]), skill_pct=float(r["test_mae_improvement_pct"]))
        if len(pers):
            out["persistence_test_mae"] = float(pers.iloc[0]["test_mae"])
        out["best_single_test"] = mc[~mc["model"].isin(["persistence", "ensemble"])].sort_values("test_mae").iloc[0]["model"]
    return out
