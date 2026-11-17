"""Meteorological data from Open-Meteo (no API key; free for non-commercial use).

Why: recent studies show that weather information strongly improves PM2.5
forecasts (e.g. Domalatov et al., 2025, Almaty: Random Forest PM2.5 RMSE
6.22 -> 3.96 ug/m3 when meteorology was added). Pollution episodes in Astana
are driven by cold, calm, stable weather that past pollution alone cannot
anticipate.

Sources (Zippenfenig, 2023, https://doi.org/10.5281/zenodo.7970649):
* Historical: https://archive-api.open-meteo.com/v1/archive (ERA5-based reanalysis)
* Operational: https://api.open-meteo.com/v1/forecast (numerical weather forecast)

Two feature groups (both computed by :func:`compute_weather_features`, used
identically for training and live inference):

* ``wx_*``  PAST weather, from hours labelled before the 18:00 issue time on
  day d (strictly leakage-free).
* ``wxf_*`` TARGET-DAY weather for local day d+1. At 18:00 such values are
  available operationally only as a weather FORECAST. For the historical
  benchmark they are taken from reanalysis ("perfect prognosis"), which is
  the standard approach when archived forecasts are not available but gives
  an optimistic upper bound. Live forecasts use the Open-Meteo forecast.
  The configuration ``full_weather_past`` (no ``wxf_*``) is reported as the
  strictly operational, forecast-free benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .config import Config, get_logger

HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "precipitation",
               "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
SHORT = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "surface_pressure": "pressure",
         "precipitation": "precip", "cloud_cover": "cloud", "wind_speed_10m": "wind",
         "wind_direction_10m": "wind_dir"}

WEATHER_PAST_FEATURES = ["wx_temp_mean_24h", "wx_rh_mean_24h", "wx_wind_mean_24h", "wx_cloud_mean_24h",
                         "wx_precip_sum_24h", "wx_pressure_last", "wx_pressure_change_24h"]
WEATHER_TARGET_FEATURES = ["wxf_temp_mean", "wxf_temp_min", "wxf_hdd", "wxf_wind_mean", "wxf_calm_hours",
                           "wxf_rh_mean", "wxf_precip_sum", "wxf_cloud_mean", "wxf_pressure_mean",
                           "wxf_temp_change", "wxf_pressure_change"]
WEATHER_FEATURES = WEATHER_PAST_FEATURES + WEATHER_TARGET_FEATURES


class WeatherError(RuntimeError):
    """Raised when weather data cannot be obtained or parsed."""


# --------------------------------------------------------------------------- parsing
def parse_open_meteo(payload: Any) -> pd.DataFrame:
    """Open-Meteo JSON (timezone=GMT) -> hourly DataFrame indexed by UTC time."""
    if not isinstance(payload, dict) or "hourly" not in payload or "time" not in payload["hourly"]:
        reason = payload.get("reason") if isinstance(payload, dict) else None
        raise WeatherError(f"Unexpected Open-Meteo response{': ' + str(reason) if reason else ''}")
    h = payload["hourly"]
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(h["time"], utc=True)})
    for v in HOURLY_VARS:
        df[SHORT[v]] = pd.to_numeric(pd.Series(h.get(v, [np.nan] * len(df))), errors="coerce").values
    return df


def _get(url: str, params: dict[str, Any], timeout: float = 60, retries: int = 4) -> Any:
    log = get_logger()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last = WeatherError(f"HTTP {r.status_code} from {url}")
            else:
                raise WeatherError(f"HTTP {r.status_code} from {url}: {r.text[:200]}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = WeatherError(f"Network error calling {url}: {exc}")
        if attempt < retries:
            import time
            delay = 2.0 * 2 ** attempt
            log.warning("%s -- retry in %.0fs", last, delay)
            time.sleep(delay)
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- download
def download_history(cfg: Config, start_date: str, end_date: str) -> pd.DataFrame:
    """Download hourly reanalysis weather year by year (cached JSON per year)."""
    log = get_logger()
    w = cfg["weather"]
    lat, lon = cfg["location"]["latitude"], cfg["location"]["longitude"]
    cache = cfg.path("weather_cache_dir")
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
    for year in range(s.year, e.year + 1):
        ys = max(s, pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        ye = min(e, pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
        f = cache / f"openmeteo_{ys}_{ye}.json"
        if f.exists():
            payload = json.loads(f.read_text(encoding="utf-8"))
        else:
            log.info("Downloading Open-Meteo weather %s .. %s", ys, ye)
            payload = _get(w["archive_url"], {"latitude": lat, "longitude": lon, "start_date": ys, "end_date": ye,
                                              "hourly": ",".join(HOURLY_VARS), "timezone": "GMT",
                                              "wind_speed_unit": "ms"})
            f.write_text(json.dumps(payload), encoding="utf-8")
        frames.append(parse_open_meteo(payload))
    df = pd.concat(frames).drop_duplicates("timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)
    return df


def fetch_forecast(cfg: Config, past_days: int = 5, forecast_days: int = 3) -> pd.DataFrame:
    """Recent past + forecast hourly weather for live inference."""
    w = cfg["weather"]
    payload = _get(w["forecast_url"], {"latitude": cfg["location"]["latitude"],
                                       "longitude": cfg["location"]["longitude"],
                                       "hourly": ",".join(HOURLY_VARS), "timezone": "GMT", "wind_speed_unit": "ms",
                                       "past_days": past_days, "forecast_days": forecast_days})
    return parse_open_meteo(payload)


def save_weather(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def run_weather_download(cfg: Config) -> pd.DataFrame:
    log = get_logger()
    d = cfg["download"]
    start = (pd.Timestamp(d["start_date"]) - pd.Timedelta(days=d["lead_in_days"] + 2)).strftime("%Y-%m-%d")
    end_default = (pd.Timestamp.now(tz=cfg.tz) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    end = min(pd.Timestamp(end_default), pd.Timestamp(cfg["weather"].get("end_date", end_default))).strftime("%Y-%m-%d")
    df = download_history(cfg, start, end)
    save_weather(df, cfg.path("weather_csv"))
    miss = df[[SHORT[v] for v in HOURLY_VARS]].isna().mean().round(4).to_dict()
    log.info("Weather: %d hourly rows %s .. %s, missing share %s -> %s", len(df),
             df["timestamp_utc"].min(), df["timestamp_utc"].max(), miss, cfg.path("weather_csv"))
    return df


# --------------------------------------------------------------------------- features
def _mean(a):
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def compute_weather_features(weather: pd.DataFrame, issue_times_local: Iterable[pd.Timestamp],
                             min_hours: int = 18) -> pd.DataFrame:
    """Weather features per issue time (same function for training and inference).

    Past window: hourly values labelled in [issue-24h, issue-1h]. Target-day
    window: local day d+1 (00:00-23:00 labels).
    """
    w = weather.set_index("timestamp_utc").sort_index()
    rows = []
    for issue_local in issue_times_local:
        issue_utc = issue_local.tz_convert("UTC")
        past = w.loc[issue_utc - pd.Timedelta(hours=24): issue_utc - pd.Timedelta(hours=1)]
        day = pd.Timestamp(issue_local.date()) + pd.Timedelta(days=1)
        t0 = day.tz_localize(issue_local.tz).tz_convert("UTC")
        t1 = (day + pd.Timedelta(days=1)).tz_localize(issue_local.tz).tz_convert("UTC") - pd.Timedelta(hours=1)
        tgt = w.loc[t0:t1]
        r: dict[str, Any] = {"issue_date": issue_local.strftime("%Y-%m-%d")}
        okp = len(past) >= min_hours
        r["wx_temp_mean_24h"] = _mean(past["temp"].values) if okp else np.nan
        r["wx_rh_mean_24h"] = _mean(past["rh"].values) if okp else np.nan
        r["wx_wind_mean_24h"] = _mean(past["wind"].values) if okp else np.nan
        r["wx_cloud_mean_24h"] = _mean(past["cloud"].values) if okp else np.nan
        r["wx_precip_sum_24h"] = float(np.nansum(past["precip"].values)) if okp else np.nan
        pr = past["pressure"].dropna()
        r["wx_pressure_last"] = float(pr.iloc[-1]) if len(pr) else np.nan
        r["wx_pressure_change_24h"] = float(pr.iloc[-1] - pr.iloc[0]) if len(pr) >= 2 else np.nan
        okt = len(tgt) >= min_hours
        temp = tgt["temp"].values
        r["wxf_temp_mean"] = _mean(temp) if okt else np.nan
        r["wxf_temp_min"] = float(np.nanmin(temp)) if okt and np.isfinite(temp).any() else np.nan
        r["wxf_hdd"] = max(0.0, 18.0 - r["wxf_temp_mean"]) if np.isfinite(r["wxf_temp_mean"]) else np.nan
        r["wxf_wind_mean"] = _mean(tgt["wind"].values) if okt else np.nan
        r["wxf_calm_hours"] = float((tgt["wind"] < 2.0).sum()) if okt else np.nan
        r["wxf_rh_mean"] = _mean(tgt["rh"].values) if okt else np.nan
        r["wxf_precip_sum"] = float(np.nansum(tgt["precip"].values)) if okt else np.nan
        r["wxf_cloud_mean"] = _mean(tgt["cloud"].values) if okt else np.nan
        r["wxf_pressure_mean"] = _mean(tgt["pressure"].values) if okt else np.nan
        r["wxf_temp_change"] = r["wxf_temp_mean"] - r["wx_temp_mean_24h"]
        r["wxf_pressure_change"] = r["wxf_pressure_mean"] - r["wx_pressure_last"]
        rows.append(r)
    return pd.DataFrame(rows)
