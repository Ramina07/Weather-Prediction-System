"""Bridge between the web app and the backend package ``air_quality``.

* live forecasts   -> ``air_quality.prediction.make_forecast`` (same code as the CLI)
* verification     -> real daily-mean PM2.5 from the OpenWeatherMap history endpoint
* model playground -> the saved benchmark pipelines in ``artifacts/models/benchmark``
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from . import paths, store
from .data import cache_resource

BENCHMARK_MODELS = ["ridge", "random_forest", "xgboost", "svr", "ensemble"]


# --------------------------------------------------------------------------- backend access
def backend_status() -> dict[str, Any]:
    """What is available on this machine (used to show friendly hints instead of tracebacks)."""
    st: dict[str, Any] = {"backend": False, "api_key": False, "xgboost": False, "operational_model": False}
    try:
        from air_quality.config import get_api_key, load_config

        cfg = load_config()
        st["backend"] = True
        st["api_key"] = bool(get_api_key(cfg.root, required=False))
        st["operational_model"] = (cfg.path("models_dir") / "operational" / "operational_model.joblib").exists()
        st["benchmark_model"] = (cfg.path("models_dir") / "benchmark" / "best_model.joblib").exists()
    except Exception as exc:  # noqa: BLE001
        st["error"] = str(exc)
    try:
        import xgboost  # noqa: F401

        st["xgboost"] = True
    except Exception:  # noqa: BLE001
        pass
    return st


def _cfg():
    from air_quality.config import load_config

    return load_config()


def _friendly(exc: Exception) -> str:
    name = type(exc).__name__
    msg = str(exc)
    hints = {
        "MissingAPIKeyError": "No OpenWeatherMap API key found. Copy `.env.example` to `.env` in the project folder "
                              "and put your key after `OPENWEATHER_API_KEY=`, then restart the app.",
        "APIAuthError": "The API key was rejected (HTTP 401). New keys can take up to 2 hours to activate.",
        "APIRateLimitError": "Too many requests to OpenWeatherMap. Wait a minute and try again.",
        "APIConnectionError": "Could not reach api.openweathermap.org. Check your internet connection.",
        "APITimeoutError": "OpenWeatherMap did not answer in time. Try again.",
        "ModelArtifactError": "No trained model found. Run `python -m air_quality.cli train` first.",
    }
    if name in hints:
        return hints[name]
    if isinstance(exc, ImportError) and "xgboost" in msg:
        return "XGBoost is not installed (the selected ensemble needs it): `pip install -r requirements.txt`."
    return f"{name}: {msg}"


class _CapturingPipeline:
    """Wraps the saved pipeline and remembers the exact feature row the backend predicts on."""

    def __init__(self, inner, store_: dict):
        self._inner, self._store = inner, store_

    def predict(self, X):
        self._store["X"] = X.copy()
        self._store["pipeline"] = self._inner
        return self._inner.predict(X)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _explain_live(captured: dict, result: dict[str, Any]) -> dict[str, Any]:
    """Model votes and key inputs for the exact feature row used by the live forecast."""
    X = captured.get("X")
    if X is None:
        return {}
    out: dict[str, Any] = {"n_features": int(X.shape[1])}
    row = X.iloc[0]
    keys = ["pm25_last", "pm25_mean_24h", "pm25_max_24h", "wxf_temp_mean", "wxf_wind_mean", "wxf_calm_hours",
            "wxf_precip_sum", "target_is_heating_season"]
    out["inputs"] = {k: (float(row[k]) if k in row.index and pd.notna(row[k]) else None) for k in keys}
    # votes of the ensemble members (the production model)
    pipe = captured.get("pipeline")
    est = pipe[-1] if hasattr(pipe, "steps") else pipe
    votes = {}
    Xf = X.astype(float)
    if hasattr(pipe, "steps") and len(pipe.steps) > 1:
        Xf = pipe[:-1].transform(Xf)
    for (n, _), e in zip(getattr(est, "estimators", []), getattr(est, "estimators_", [])):
        try:
            votes[n] = float(np.clip(e.predict(Xf)[0], 0, None))
        except Exception:  # noqa: BLE001
            continue
    out["members"] = votes
    # second opinions from the individual benchmark models (trained on 2021-2023)
    others = {}
    for name, b in load_benchmark_models().items():
        if name == "ensemble":
            continue
        try:
            if set(b["features"]) <= set(X.columns):
                others[name] = float(np.clip(b["pipeline"].predict(X[b["features"]].astype(float))[0], 0, None))
        except Exception:  # noqa: BLE001
            continue
    if result.get("recent_pm25_mean_24h") is not None:
        others["persistence"] = float(result["recent_pm25_mean_24h"])
    out["others"] = others
    return out


def run_live_forecast(issue_mode: str = "auto", model: str = "auto", with_provider: bool = True,
                      force: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    """Run the real backend forecast. ``issue_mode='auto'`` picks strict after 18:00 and latest before."""
    try:
        import air_quality.prediction as P
        from air_quality.prediction import ForecastTimingError, make_forecast

        captured: dict = {}
        original = P.load_bundle

        def _patched(cfg_, which="auto"):
            b, p = original(cfg_, which)
            b = dict(b)
            b["pipeline"] = _CapturingPipeline(b["pipeline"], captured)
            return b, p

        cfg = _cfg()
        mode = issue_mode
        if issue_mode == "auto":
            now_local = pd.Timestamp.now(tz=cfg.tz)
            mode = "strict" if now_local.hour >= cfg["forecast"]["issue_hour_local"] else "latest"
        P.load_bundle = _patched
        try:
            try:
                res = make_forecast(cfg, issue_mode=mode, model=model, with_provider=with_provider, force=force)
            except ForecastTimingError:
                res = make_forecast(cfg, issue_mode="latest", model=model, with_provider=with_provider, force=force)
        finally:
            P.load_bundle = original
        store.save_forecast(res, kind="live")
        try:
            res = dict(res)
            res["explain"] = _explain_live(captured, res)
        except Exception:  # noqa: BLE001
            pass
        return res, None
    except Exception as exc:  # noqa: BLE001
        return None, _friendly(exc)


def run_hindcast(days_back: int = 3, model: str = "auto") -> tuple[list[dict[str, Any]], list[str]]:
    """Re-create the forecasts that *would have been* issued at 18:00 on each of the last ``days_back`` days.

    Uses only data available at each issue time from the pollution API. Tomorrow's-weather features come from
    the Open-Meteo recent-past series (analysis rather than a true archived forecast), so hindcasts are labelled
    separately from live forecasts.
    """
    results, errors = [], []
    try:
        from air_quality.prediction import make_forecast
        from air_quality.weather import fetch_forecast

        cfg = _cfg()
        tz, hour = cfg.tz, cfg["forecast"]["issue_hour_local"]
        try:
            wx = fetch_forecast(cfg, past_days=min(7, days_back + 3), forecast_days=2)
            weather_source: Callable | None = (lambda _cfg: wx)
        except Exception as exc:  # noqa: BLE001
            weather_source = None
            errors.append(f"Weather unavailable ({exc}); the no-weather fallback model is used.")
        today = pd.Timestamp.now(tz=tz).normalize()
        for d in range(days_back, 0, -1):
            issue_day = today - pd.Timedelta(days=d)
            now = (issue_day + pd.Timedelta(hours=hour, minutes=5)).tz_convert("UTC")
            try:
                r = make_forecast(cfg, now=now, issue_mode="strict", model=model, with_provider=False,
                                  force=True, save=False, weather_source=weather_source)
                r["label"] = "Hindcast (retrospective)"
                store.save_forecast(r, kind="hindcast")
                results.append(r)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{issue_day:%Y-%m-%d}: {_friendly(exc)}")
    except Exception as exc:  # noqa: BLE001
        errors.append(_friendly(exc))
    return results, errors


def fetch_daily_actual(date_str: str, client=None, cfg=None) -> tuple[float | None, int]:
    """Mean of the hourly PM2.5 records of one local calendar day (None if fewer than 20 valid hours)."""
    from air_quality.api_client import OpenWeatherAirPollutionClient
    from air_quality.config import get_api_key

    cfg = cfg or _cfg()
    tz = cfg.tz
    if client is None:
        client = OpenWeatherAirPollutionClient.from_config(cfg, get_api_key(cfg.root))
    lat, lon = cfg["location"]["latitude"], cfg["location"]["longitude"]
    day = pd.Timestamp(date_str).tz_localize(tz)
    start, end = int(day.tz_convert("UTC").timestamp()), int((day + pd.Timedelta(days=1)).tz_convert("UTC").timestamp())
    hist = client.history(lat, lon, start, end - 1)
    vals = []
    for rec in hist.records:
        ts = pd.Timestamp(rec["dt"], unit="s", tz="UTC").tz_convert(tz)
        v = rec.get("pm2_5")
        if ts.strftime("%Y-%m-%d") == date_str and v is not None and np.isfinite(v):
            vals.append(float(v))
    n = len(vals)
    min_h = cfg["forecast"]["target_min_valid_hours"]
    return (float(np.mean(vals)) if n >= min_h else None), n


def verify_pending() -> tuple[int, list[str]]:
    """Download the real outcome for every finished target day that is still unverified."""
    msgs: list[str] = []
    try:
        cfg = _cfg()
        today = pd.Timestamp.now(tz=cfg.tz).strftime("%Y-%m-%d")
        dates = store.pending_dates(today)
        if not dates:
            return 0, ["Nothing to verify: every finished day already has its real value."]
        from air_quality.api_client import OpenWeatherAirPollutionClient
        from air_quality.config import get_api_key

        client = OpenWeatherAirPollutionClient.from_config(cfg, get_api_key(cfg.root))
        done = 0
        for d in dates:
            try:
                actual, n = fetch_daily_actual(d, client=client, cfg=cfg)
                store.save_observation(d, actual, n)
                if actual is None:
                    msgs.append(f"{d}: only {n} valid hours so far (need 20). Will retry later.")
                else:
                    done += 1
            except Exception as exc:  # noqa: BLE001
                msgs.append(f"{d}: {_friendly(exc)}")
        store.log_event("verify", f"{done} day(s) verified")
        return done, msgs
    except Exception as exc:  # noqa: BLE001
        return 0, [_friendly(exc)]


# --------------------------------------------------------------------------- model playground
@cache_resource
def load_benchmark_models() -> dict[str, dict[str, Any]]:
    """Load the saved benchmark pipelines (train-only fits). Models that cannot be loaded are skipped."""
    import joblib

    out: dict[str, dict[str, Any]] = {}
    for name in BENCHMARK_MODELS:
        p = paths.MODELS / "benchmark" / f"{name}.joblib"
        if not p.exists():
            continue
        try:
            b = joblib.load(p)
            if isinstance(b, dict) and "pipeline" in b:
                out[name] = b
        except Exception:  # noqa: BLE001  (e.g. xgboost not installed)
            continue
    return out


def predict_all(models: dict[str, dict[str, Any]], X: pd.DataFrame) -> pd.DataFrame:
    """Predictions of every loaded model for each row of X (columns = model names)."""
    res = {}
    for name, b in models.items():
        try:
            res[name] = np.clip(b["pipeline"].predict(X[b["features"]].astype(float)), 0, None)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(res, index=X.index)


def ensemble_members(models: dict[str, dict[str, Any]], X: pd.DataFrame) -> dict[str, float]:
    """Individual votes inside the VotingRegressor (shows how the ensemble averages)."""
    b = models.get("ensemble")
    if not b:
        return {}
    pipe = b["pipeline"]
    est = pipe[-1] if hasattr(pipe, "steps") else pipe
    names = [n for n, _ in getattr(est, "estimators", [])]
    fitted = getattr(est, "estimators_", [])
    out = {}
    Xf = X[b["features"]].astype(float)
    if hasattr(pipe, "steps") and len(pipe.steps) > 1:
        Xf = pipe[:-1].transform(Xf)
    for n, e in zip(names, fitted):
        try:
            out[n] = float(np.clip(e.predict(Xf)[0], 0, None))
        except Exception:  # noqa: BLE001
            continue
    return out


PM_LEVEL_COLS = ["pm25_last", "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_72h", "pm25_mean_6h", "pm25_mean_12h",
                 "pm25_mean_24h", "pm25_mean_48h", "pm25_mean_72h", "pm25_std_24h", "pm25_min_24h",
                 "pm25_max_24h", "pm25_diff_24h", "pm25_trend_24h", "pm25_prev_day_mean", "pm10_last",
                 "pm10_lag_24h", "pm10_mean_24h"]


def apply_scenario(base: pd.Series, pm_level: float | None = None, temp: float | None = None,
                   wind: float | None = None, calm_hours: float | None = None,
                   precip: float | None = None, heating: bool | None = None) -> pd.DataFrame:
    """Return a one-row frame with a what-if scenario applied consistently to related features."""
    r = base.copy()
    if pm_level is not None:
        cur = float(r.get("pm25_mean_24h", np.nan))
        if np.isfinite(cur) and cur > 0:
            k = pm_level / cur
            for c in PM_LEVEL_COLS:
                if c in r.index and pd.notna(r[c]):
                    r[c] = float(r[c]) * k
        else:
            for c in PM_LEVEL_COLS:
                if c.startswith("pm25_mean") or c in ("pm25_last", "pm25_prev_day_mean"):
                    r[c] = pm_level
    if temp is not None:
        old = float(r.get("wxf_temp_mean", np.nan))
        r["wxf_temp_mean"] = temp
        if np.isfinite(old) and pd.notna(r.get("wxf_temp_min")):
            r["wxf_temp_min"] = float(r["wxf_temp_min"]) + (temp - old)
        else:
            r["wxf_temp_min"] = temp - 4
        r["wxf_hdd"] = max(0.0, 18.0 - temp)
        if pd.notna(r.get("wx_temp_mean_24h")):
            r["wxf_temp_change"] = temp - float(r["wx_temp_mean_24h"])
    if wind is not None:
        r["wxf_wind_mean"] = wind
    if calm_hours is not None:
        r["wxf_calm_hours"] = calm_hours
    if precip is not None:
        r["wxf_precip_sum"] = precip
    if heating is not None:
        r["target_is_heating_season"] = float(heating)
    return pd.DataFrame([r])


SWEEPS = {
    "pm_level": ("Recent PM2.5 level (last-24 h mean, µg/m³)", 0.5, 45.0),
    "temp": ("Tomorrow's mean temperature (°C)", -35.0, 30.0),
    "wind": ("Tomorrow's mean wind speed (m/s)", 0.0, 12.0),
    "calm_hours": ("Calm hours tomorrow (wind < 2 m/s)", 0.0, 24.0),
}


def sweep(models: dict[str, dict[str, Any]], base: pd.Series, var: str, n: int = 31,
          **fixed: Any) -> pd.DataFrame:
    """Prediction of every model while one input varies and everything else is held fixed."""
    _, lo, hi = SWEEPS[var]
    grid = np.linspace(lo, hi, n)
    frames = []
    for g in grid:
        kw = dict(fixed)
        kw[var] = float(g)
        frames.append(apply_scenario(base, **kw))
    X = pd.concat(frames, ignore_index=True)
    P = predict_all(models, X)
    P.insert(0, var, grid)
    return P
