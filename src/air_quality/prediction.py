"""Live next-day PM2.5 forecast ("Our ML Forecast").

Steps: read key -> fetch the latest hourly history -> clean with the same
function as training -> compute features with the same function as training
-> check freshness/completeness -> load the saved pipeline -> predict ->
print and save JSON + CSV.

Issue-time protocol: the model was trained for an 18:00 local issue time.
If the command runs before 18:00 the default ``strict`` mode refuses (the
horizon would otherwise change silently). ``--issue-mode latest`` explicitly
uses the most recent completed issue time (yesterday 18:00), whose target is
TODAY; this is printed in the output.

The OpenWeatherMap provider forecast can optionally be shown, always
labelled "OpenWeatherMap Provider Forecast" and never as our model output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .api_client import OpenWeatherAirPollutionClient
from .config import Config, get_api_key, get_logger
from .feature_engineering import compute_features, issue_time_for_date
from .preprocessing import clean_hourly
from .weather import WeatherError, compute_weather_features, fetch_forecast


class ForecastTimingError(RuntimeError):
    """Raised when a strict-mode forecast is requested before the issue hour."""


class StaleDataError(RuntimeError):
    """Raised when live data are too old or too incomplete for a reliable forecast."""


class ModelArtifactError(FileNotFoundError):
    """Raised when no saved model artifact is available."""


def load_bundle(cfg: Config, which: str = "auto") -> tuple[dict[str, Any], Path]:
    mdir = cfg.path("models_dir")
    op, bm = mdir / "operational" / "operational_model.joblib", mdir / "benchmark" / "best_model.joblib"
    order = {"auto": [op, bm], "operational": [op], "benchmark": [bm]}.get(which)
    if order is None:
        raise ValueError("model must be one of: auto, operational, benchmark")
    for p in order:
        if p.exists():
            b = joblib.load(p)
            if not isinstance(b, dict) or "pipeline" not in b or "features" not in b:
                raise ModelArtifactError(f"{p} is not a valid model bundle")
            return b, p
    raise ModelArtifactError("No trained model found (looked for: " + ", ".join(str(p) for p in order)
                             + "). Run `python -m air_quality.cli train` first.")


def resolve_issue_time(now_local: pd.Timestamp, tz: str, issue_hour: int, mode: str) -> tuple[pd.Timestamp, str]:
    today_issue = issue_time_for_date(now_local.tz_localize(None).normalize(), tz, issue_hour)
    if now_local >= today_issue:
        return today_issue, "standard: issued today at the operational issue hour; target = tomorrow"
    if mode == "strict":
        raise ForecastTimingError(
            f"It is {now_local:%H:%M} local time; the model's protocol issues forecasts at {issue_hour:02d}:00. "
            f"Re-run after {issue_hour:02d}:00 for tomorrow's forecast, or use `--issue-mode latest` to obtain the "
            f"forecast issued yesterday at {issue_hour:02d}:00 for TODAY.")
    if mode == "latest":
        return today_issue - pd.Timedelta(days=1), (
            "latest: issued at the most recent completed issue time (yesterday); target = TODAY")
    raise ValueError("issue mode must be 'strict' or 'latest'")


def records_to_raw(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df.rename(columns={"dt": "dt_unix"})


def make_forecast(cfg: Config, client: OpenWeatherAirPollutionClient | None = None,
                  now: pd.Timestamp | None = None, issue_mode: str = "strict", model: str = "auto",
                  with_provider: bool = False, force: bool = False, save: bool = True,
                  weather_source=None) -> dict[str, Any]:
    log = get_logger()
    fc = cfg["forecast"]
    tz = cfg.tz
    bundle, bpath = load_bundle(cfg, model)
    now_local = (now or pd.Timestamp.now(tz="UTC")).tz_convert(tz)
    issue_local, protocol_note = resolve_issue_time(now_local, tz, fc["issue_hour_local"], issue_mode)
    issue_utc = issue_local.tz_convert("UTC")
    if client is None:
        client = OpenWeatherAirPollutionClient.from_config(cfg, get_api_key(cfg.root))
    lat, lon = cfg["location"]["latitude"], cfg["location"]["longitude"]
    start = int((issue_utc - pd.Timedelta(hours=fc["live_history_hours"])).timestamp())
    end = int(min(now_local.tz_convert("UTC"), issue_utc + pd.Timedelta(hours=1)).timestamp())
    hist = client.history(lat, lon, start, end)
    if not hist.records:
        raise StaleDataError("The history endpoint returned no observations for the required window.")
    raw = records_to_raw(hist.records)
    grid, _ = clean_hourly(raw, tz, now_utc=now_local.tz_convert("UTC"))
    feats = compute_features(grid, [issue_local], fc, cfg["evaluation"]["heating_season"])
    row = feats.iloc[0]

    warnings: list[str] = []
    weather_note = "not used by this model"
    needs_weather = any(f.startswith(("wx_", "wxf_")) for f in bundle["features"])
    if needs_weather:
        try:
            wdf = weather_source(cfg) if weather_source else fetch_forecast(cfg)
            wf = compute_weather_features(wdf, [issue_local])
            if wf[[f for f in bundle["features"] if f.startswith(("wx_", "wxf_"))]].isna().all(axis=None):
                raise WeatherError("weather forecast returned no usable values for the required window")
            for c in wf.columns:
                if c != "issue_date":
                    feats[c] = wf[c].values
            weather_note = "Open-Meteo forecast (past 24 h + forecast for the target day)"
        except Exception as exc:  # fall back to the model trained without weather
            fb = cfg.path("models_dir") / "benchmark" / "best_model_no_weather.joblib"
            if cfg["weather"].get("required_for_prediction") or not fb.exists():
                raise StaleDataError(f"Weather forecast unavailable ({exc}) and no fallback model.") from None
            bundle, bpath = joblib.load(fb), fb
            warnings_fallback = f"Weather forecast unavailable ({exc}); used the model without weather features."
            log.warning(warnings_fallback)
            weather_note = "unavailable -> fallback model without weather"
    newest_label = pd.Timestamp(row["meta_newest_record_utc"]) if row["meta_newest_record_utc"] else None
    newest_obs = grid.loc[grid["is_observed"] & (grid["timestamp_utc"] < issue_utc), "timestamp_utc"].max()
    stale_h = ((issue_utc - pd.Timedelta(hours=1) - newest_obs) / pd.Timedelta(hours=1)) if pd.notna(newest_obs) else np.inf
    if stale_h > fc["max_staleness_hours"]:
        warnings.append(f"Newest usable observation is {stale_h:.0f} h older than the newest possible record "
                        f"(limit {fc['max_staleness_hours']} h).")
    for f in ("pm25_last", "pm25_mean_24h"):
        if not np.isfinite(row[f]):
            warnings.append(f"Required feature {f} could not be computed (insufficient recent data).")
    if row["meta_pm25_coverage_72h"] < fc["rolling_min_coverage"]:
        warnings.append(f"Only {100 * row['meta_pm25_coverage_72h']:.0f}% of the last 72 h have PM2.5 values.")
    if warnings and not force:
        raise StaleDataError("Live data are not sufficient for a reliable forecast: " + " ".join(warnings)
                             + " Use --force to produce a flagged forecast anyway.")

    X = feats[bundle["features"]]
    missing_cols = [c for c in bundle["features"] if c not in feats.columns]
    if missing_cols:  # pragma: no cover - schema guaranteed by compute_features
        raise ModelArtifactError(f"Feature schema mismatch: {missing_cols}")
    pred = float(np.clip(bundle["pipeline"].predict(X)[0], 0, None))
    ri = bundle.get("residual_interval") or {}
    interval = None
    if np.isfinite(ri.get("lower", np.nan)):
        interval = {"coverage_nominal": ri["coverage"], "lower": max(0.0, pred + ri["lower"]),
                    "upper": pred + ri["upper"], "method": ri.get("source")}
    target_date = (issue_local.tz_localize(None).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    result: dict[str, Any] = {
        "label": "Our ML Forecast",
        "location": cfg["location"]["name"],
        "coordinates": {"lat": lat, "lon": lon},
        "forecast_issued_local": issue_local.isoformat(),
        "forecast_generated_at_local": now_local.isoformat(),
        "issue_protocol": protocol_note,
        "prediction_date": target_date,
        "target": "mean PM2.5 over local calendar day (00:00-23:59, Asia/Almaty)",
        "predicted_pm25": round(pred, 2),
        "units": "µg/m³",
        "prediction_interval": interval,
        "model": bundle["model_name"],
        "feature_config": bundle["feature_config"],
        "model_artifact": str(bpath.relative_to(cfg.root)) if bpath.is_relative_to(cfg.root) else str(bpath),
        "model_fitted_on": bundle.get("fitted_on"),
        "data_freshness_latest_observation_utc": newest_obs.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(newest_obs) else None,
        "data_freshness_latest_observation_local": newest_obs.tz_convert(tz).isoformat() if pd.notna(newest_obs) else None,
        "pm25_coverage_last_72h_pct": round(100 * float(row["meta_pm25_coverage_72h"]), 1),
        "recent_pm25_mean_24h": round(float(row["pm25_mean_24h"]), 2) if np.isfinite(row["pm25_mean_24h"]) else None,
        "weather_data": weather_note,
        "reliable": not warnings,
        "warnings": warnings,
        "api_requests": client.n_requests,
        "data_source": "OpenWeatherMap Air Pollution API (history endpoint)",
    }
    if with_provider:
        try:
            pf = client.forecast(lat, lon)
            p = pd.DataFrame(pf.records)
            p["date_local"] = pd.to_datetime(p["dt"], unit="s", utc=True).dt.tz_convert(tz).dt.strftime("%Y-%m-%d")
            sel = p[p["date_local"] == target_date]["pm2_5"].dropna()
            result["provider_forecast"] = {
                "label": "OpenWeatherMap Provider Forecast",
                "note": "Published by the provider; NOT produced by our ML model.",
                "mean_pm25_target_date": round(float(sel.mean()), 2) if len(sel) else None,
                "n_hours": int(len(sel))}
        except Exception as exc:  # provider forecast is optional
            result["provider_forecast"] = {"label": "OpenWeatherMap Provider Forecast", "error": str(exc)}
    if save:
        save_forecast(cfg, result)
    log.info("Forecast for %s: %.2f µg/m³ (%s)", target_date, pred, bundle["model_name"])
    return result


def save_forecast(cfg: Config, result: dict[str, Any]) -> tuple[Path, Path]:
    pdir = cfg.path("predictions_dir")
    pdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp = pdir / f"forecast_{result['prediction_date']}_{stamp}.json"
    jp.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (pdir / "latest_forecast.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    flat = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
    if result.get("prediction_interval"):
        flat["interval_lower"] = result["prediction_interval"]["lower"]
        flat["interval_upper"] = result["prediction_interval"]["upper"]
    flat["warnings"] = " | ".join(result["warnings"])
    if result.get("provider_forecast"):
        flat["provider_forecast_mean_pm25"] = result["provider_forecast"].get("mean_pm25_target_date")
    cp = pdir / "forecast_history.csv"
    pd.DataFrame([flat]).to_csv(cp, mode="a", header=not cp.exists(), index=False)
    return jp, cp


def format_forecast(r: dict[str, Any]) -> str:
    lines = ["", "=" * 64, f"  {r['label']} — next-day mean PM2.5", "=" * 64,
             f"Forecast location : {r['location']} ({r['coordinates']['lat']}, {r['coordinates']['lon']})",
             f"Forecast issued   : {r['forecast_issued_local']}",
             f"Protocol          : {r['issue_protocol']}",
             f"Prediction date   : {r['prediction_date']}",
             f"Selected model    : {r['model']} [{r['feature_config']}] ({r['model_fitted_on']})",
             f"Predicted PM2.5   : {r['predicted_pm25']:.2f} {r['units']}"]
    if r.get("prediction_interval"):
        pi = r["prediction_interval"]
        lines.append(f"Approx. {int(100 * pi['coverage_nominal'])}% interval: "
                     f"{pi['lower']:.2f} – {pi['upper']:.2f} {r['units']} (empirical, validation residuals)")
    lines += [f"Data freshness    : latest observation {r['data_freshness_latest_observation_local']}",
              f"PM2.5 coverage 72h: {r['pm25_coverage_last_72h_pct']}%",
              f"Weather inputs    : {r.get('weather_data')}",
              f"Reliable          : {'yes' if r['reliable'] else 'NO — see warnings'}"]
    for w in r["warnings"]:
        lines.append(f"  WARNING: {w}")
    if r.get("provider_forecast"):
        p = r["provider_forecast"]
        lines += ["-" * 64, f"{p['label']} (for reference only, not our model):",
                  f"  mean PM2.5 on {r['prediction_date']}: {p.get('mean_pm25_target_date')} µg/m³ "
                  f"({p.get('n_hours', 0)} hourly values)" if "error" not in p else f"  unavailable: {p['error']}"]
    lines.append("=" * 64)
    return "\n".join(lines)
