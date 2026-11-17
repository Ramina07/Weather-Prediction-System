"""Command-line interface for the complete workflow.

Usage (from the ML_final folder, with the virtual environment active)::

    python -m air_quality.cli env
    python -m air_quality.cli test-api
    python -m air_quality.cli download [--start 2021-01-01 --end 2025-12-31]
    python -m air_quality.cli weather            # Open-Meteo historical weather
    python -m air_quality.cli audit
    python -m air_quality.cli preprocess
    python -m air_quality.cli eda
    python -m air_quality.cli train [--quick]
    python -m air_quality.cli evaluate
    python -m air_quality.cli explain
    python -m air_quality.cli update              # download 2026 data for operational use
    python -m air_quality.cli train-operational
    python -m air_quality.cli predict [--issue-mode strict|latest] [--model auto|benchmark|operational] [--with-provider]
    python -m air_quality.cli report
    python -m air_quality.cli run-all [--quick] [--skip-download]

Exit codes: 0 success, 1 unexpected error, 2 configuration error, 3 missing
API key, 4 API error, 5 data error / stale data, 6 missing model artifact,
7 forecast requested outside the issue-time protocol.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXIT_OK, EXIT_ERR, EXIT_CONFIG, EXIT_KEY, EXIT_API, EXIT_DATA, EXIT_MODEL, EXIT_TIMING = 0, 1, 2, 3, 4, 5, 6, 7


def _cfg(args):
    from .config import load_config, setup_logging
    cfg = load_config(args.config)
    setup_logging(cfg, name="pipeline")
    cfg.ensure_dirs()
    return cfg


def _client(cfg):
    from .api_client import OpenWeatherAirPollutionClient
    from .config import get_api_key
    return OpenWeatherAirPollutionClient.from_config(cfg, get_api_key(cfg.root))


# --------------------------------------------------------------------------- commands
def cmd_env(args) -> int:
    cfg = _cfg(args)
    import importlib
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["pandas", "numpy", "requests", "dotenv", "yaml", "matplotlib", "seaborn", "sklearn", "xgboost",
              "scipy", "statsmodels", "joblib", "pytest", "nbformat", "nbconvert", "shap", "tzdata"]:
        try:
            mod = importlib.import_module(m)
            vers[m] = getattr(mod, "__version__", "installed")
        except ImportError:
            vers[m] = "NOT INSTALLED"
    p = cfg.path("logs_dir") / "environment.json"
    p.write_text(json.dumps(vers, indent=2), encoding="utf-8")
    for k, v in vers.items():
        print(f"{k:<12} {v}")
    print(f"\nSaved to {p}")
    return EXIT_OK


def cmd_test_api(args) -> int:
    cfg = _cfg(args)
    import pandas as pd
    client = _client(cfg)
    lat, lon = cfg["location"]["latitude"], cfg["location"]["longitude"]
    report: dict = {"tested_at_utc": datetime.now(timezone.utc).isoformat(), "coordinates": [lat, lon]}

    def describe(parsed, name):
        recs = parsed.records
        types = {k: type(v).__name__ for k, v in (recs[0].items() if recs else [])}
        info = {"http_status": 200, "n_observations": len(recs), "schema_ok": True,
                "pm25_present": bool(recs) and all(r.get("pm2_5") is not None for r in recs),
                "n_missing_pm25": parsed.n_missing_pm25, "field_types": types,
                "api_coordinates": [parsed.lat, parsed.lon]}
        if recs:
            ts = pd.to_datetime([r["dt"] for r in recs], unit="s", utc=True)
            info.update({"first_dt_utc": ts.min().isoformat(), "last_dt_utc": ts.max().isoformat(),
                         "example_pm2_5": recs[0].get("pm2_5")})
        print(f"[{name}] HTTP 200 | observations={len(recs)} | PM2.5 present={info['pm25_present']} | "
              f"fields={list(types)}")
        return info

    report["current"] = describe(client.current(lat, lon), "current")
    a = cfg["api"]
    s = int(pd.Timestamp(a["smoke_test_history_start_utc"]).timestamp())
    e = int(pd.Timestamp(a["smoke_test_history_end_utc"]).timestamp())
    h = describe(client.history(lat, lon, s, e), "history 2025-06-01..02 UTC")
    h["expected_hours"] = (e - s) // 3600 + 1
    report["history"] = h
    report["n_requests"] = client.n_requests
    out = cfg.path("logs_dir") / "api_smoke_test.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"API smoke test PASSED. Report: {out}")
    return EXIT_OK


def cmd_download(args) -> int:
    cfg = _cfg(args)
    from .data_validation import audit_hourly, format_audit
    from .download import run_download
    df, summary = run_download(cfg, _client(cfg), args.start, args.end, operational=args.operational)
    print(f"API requests: {summary.n_requests} | records returned (incl. cached): {summary.n_records_returned}")
    if len(df):
        print(format_audit(audit_hourly(df, cfg.tz, min_valid_hours=cfg["forecast"]["target_min_valid_hours"]),
                           "Downloaded dataset"))
    if summary.n_chunks_failed:
        print(f"WARNING: {summary.n_chunks_failed} chunk(s) failed; re-run the same command to retry:")
        for s, e, err in summary.failed_intervals:
            print(f"  {s} .. {e}: {err}")
        return EXIT_API
    return EXIT_OK


def cmd_weather(args) -> int:
    from .weather import run_weather_download
    run_weather_download(_cfg(args))
    return EXIT_OK


def cmd_audit(args) -> int:
    cfg = _cfg(args)
    import pandas as pd
    from .data_validation import audit_hourly, format_audit
    p = cfg.path("raw_csv")
    if not p.exists():
        print(f"Raw dataset not found: {p}. Run `download` first.")
        return EXIT_DATA
    a = audit_hourly(pd.read_csv(p), cfg.tz, min_valid_hours=cfg["forecast"]["target_min_valid_hours"])
    print(format_audit(a, "Raw dataset audit"))
    (cfg.path("metrics_dir") / "dataset_audit.json").write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
    return EXIT_OK


def cmd_preprocess(args) -> int:
    cfg = _cfg(args)
    from .feature_engineering import run_features
    from .preprocessing import run_preprocess
    run_preprocess(cfg)
    run_features(cfg)
    return EXIT_OK


def cmd_eda(args) -> int:
    from .eda import run_eda
    run_eda(_cfg(args))
    return EXIT_OK


def cmd_train(args) -> int:
    cfg = _cfg(args)
    from .training import run_training
    out = run_training(cfg, quick=args.quick)
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out["comparison"][["model", "feature_config", "validation_mae", "test_mae", "test_rmse", "test_r2",
                                 "test_mae_improvement_pct"]].round(3).to_string(index=False))
    return EXIT_OK


def cmd_evaluate(args) -> int:
    from .evaluation import run_evaluation
    run_evaluation(_cfg(args))
    return EXIT_OK


def cmd_explain(args) -> int:
    from .explainability import run_explainability
    run_explainability(_cfg(args))
    return EXIT_OK


def cmd_update(args) -> int:
    cfg = _cfg(args)
    from .download import run_download
    from .feature_engineering import run_features
    from .preprocessing import run_preprocess
    run_download(cfg, _client(cfg), operational=True)
    run_preprocess(cfg, operational=True)
    run_features(cfg, operational=True)
    return EXIT_OK


def cmd_train_operational(args) -> int:
    from .training import train_operational
    train_operational(_cfg(args))
    return EXIT_OK


def cmd_predict(args) -> int:
    cfg = _cfg(args)
    from .prediction import format_forecast, make_forecast
    r = make_forecast(cfg, issue_mode=args.issue_mode, model=args.model, with_provider=args.with_provider,
                      force=args.force)
    print(format_forecast(r))
    print(f"Saved: {cfg.path('predictions_dir') / 'latest_forecast.json'} and forecast_history.csv")
    return EXIT_OK


def cmd_report(args) -> int:
    from .report import build_report
    p = build_report(_cfg(args))
    print(f"Report written to {p}")
    return EXIT_OK


def cmd_run_all(args) -> int:
    cfg = _cfg(args)
    steps = []
    if not args.skip_download:
        steps += [("test-api", cmd_test_api), ("download", cmd_download), ("weather", cmd_weather)]
    steps += [("audit", cmd_audit), ("preprocess", cmd_preprocess), ("eda", cmd_eda)]
    if args.skip_training and (cfg.path("models_dir") / "benchmark" / "best_model.joblib").exists():
        print("Skipping training (existing benchmark model found).")
    else:
        steps.append(("train", cmd_train))
    steps += [("evaluate", cmd_evaluate), ("explain", cmd_explain), ("report", cmd_report)]
    for name, fn in steps:
        t0 = time.perf_counter()
        print(f"\n>>> {name}")
        rc = fn(args)
        print(f"<<< {name} finished in {time.perf_counter() - t0:.1f}s (exit {rc})")
        if rc != EXIT_OK:
            return rc
    return EXIT_OK


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m air_quality.cli",
                                description="Next-day PM2.5 forecasting for Astana (OpenWeatherMap data).")
    p.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="record Python/package versions").set_defaults(func=cmd_env)
    sub.add_parser("test-api", help="API smoke test (current + 2025-06-01..02 history)").set_defaults(func=cmd_test_api)
    d = sub.add_parser("download", help="download historical data (resumable)")
    d.add_argument("--start", default=None, help="local start date YYYY-MM-DD (default from config)")
    d.add_argument("--end", default=None, help="local end date YYYY-MM-DD (default from config)")
    d.add_argument("--operational", action="store_true", help="download the operational (2026+) period")
    d.set_defaults(func=cmd_download)
    sub.add_parser("weather", help="download Open-Meteo historical weather (no key needed)").set_defaults(func=cmd_weather)
    sub.add_parser("audit", help="audit the raw dataset").set_defaults(func=cmd_audit)
    sub.add_parser("preprocess", help="clean data and build the feature dataset").set_defaults(func=cmd_preprocess)
    sub.add_parser("eda", help="exploratory figures and statistics").set_defaults(func=cmd_eda)
    t = sub.add_parser("train", help="tune, validate and test the 4 models + persistence")
    t.add_argument("--quick", action="store_true", help="small search budget (smoke runs)")
    t.set_defaults(func=cmd_train)
    sub.add_parser("evaluate", help="experiments E1-E5, E7 tables and figures").set_defaults(func=cmd_evaluate)
    sub.add_parser("explain", help="experiment E6 interpretability").set_defaults(func=cmd_explain)
    sub.add_parser("update", help="download + preprocess the operational 2026 period").set_defaults(func=cmd_update)
    sub.add_parser("train-operational", help="refit best config on all labelled data").set_defaults(func=cmd_train_operational)
    pr = sub.add_parser("predict", help="forecast tomorrow's mean PM2.5")
    pr.add_argument("--issue-mode", choices=["strict", "latest"], default="strict")
    pr.add_argument("--model", choices=["auto", "benchmark", "operational"], default="auto")
    pr.add_argument("--with-provider", action="store_true", help="also show the OpenWeatherMap provider forecast")
    pr.add_argument("--force", action="store_true", help="return a flagged forecast even if data are stale")
    pr.set_defaults(func=cmd_predict)
    sub.add_parser("report", help="regenerate docs/research_report_draft.md from results").set_defaults(func=cmd_report)
    ra = sub.add_parser("run-all", help="full pipeline in order")
    ra.add_argument("--quick", action="store_true")
    ra.add_argument("--skip-download", action="store_true", help="use existing raw data")
    ra.add_argument("--skip-training", action="store_true", help="reuse existing benchmark model if present")
    ra.add_argument("--start", default=None)
    ra.add_argument("--end", default=None)
    ra.set_defaults(func=cmd_run_all, operational=False)
    return p


def main(argv: list[str] | None = None) -> int:
    from .api_client import APIError
    from .config import ConfigError, MissingAPIKeyError, redact
    from .data_validation import DataValidationError
    from .prediction import ForecastTimingError, ModelArtifactError, StaleDataError
    from .weather import WeatherError
    for stream in (sys.stdout, sys.stderr):  # Windows consoles/pipes: never crash on 'µg/m³'
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"CONFIGURATION ERROR: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except MissingAPIKeyError as e:
        print(f"MISSING API KEY: {e}", file=sys.stderr)
        return EXIT_KEY
    except APIError as e:
        print(f"API ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return EXIT_API
    except ForecastTimingError as e:
        print(f"FORECAST TIMING: {e}", file=sys.stderr)
        return EXIT_TIMING
    except ModelArtifactError as e:
        print(f"MODEL ARTIFACT ERROR: {e}", file=sys.stderr)
        return EXIT_MODEL
    except WeatherError as e:
        print(f"WEATHER DATA ERROR: {e}", file=sys.stderr)
        return EXIT_API
    except (StaleDataError, DataValidationError, FileNotFoundError, ValueError) as e:
        print(f"DATA ERROR: {redact(str(e))}", file=sys.stderr)
        return EXIT_DATA
    except KeyboardInterrupt:
        print("Interrupted. Downloads can be resumed by re-running the same command.", file=sys.stderr)
        return EXIT_ERR
    except Exception as e:  # noqa: BLE001
        print(f"UNEXPECTED ERROR ({type(e).__name__}): {redact(str(e))}", file=sys.stderr)
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
