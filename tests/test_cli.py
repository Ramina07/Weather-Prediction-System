"""CLI and end-to-end tests on SYNTHETIC fixtures (never real results)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from air_quality import cli
from air_quality.api_client import APIAuthError, ParsedResponse
from air_quality.config import ConfigError, load_config
from air_quality.feature_engineering import run_features
from air_quality.prediction import ForecastTimingError, StaleDataError, make_forecast
from air_quality.preprocessing import run_preprocess
from air_quality.training import run_training

from .conftest import have_xgboost, synthetic_raw, synthetic_weather
from air_quality.weather import save_weather


class FakeClient:
    """Stand-in API client serving SYNTHETIC records."""

    def __init__(self, raw: pd.DataFrame, error: Exception | None = None):
        self.raw = raw
        self.error = error
        self.n_requests = 0

    def history(self, lat, lon, start, end):
        self.n_requests += 1
        if self.error:
            raise self.error
        r = self.raw[(self.raw.dt_unix >= start) & (self.raw.dt_unix <= end)].rename(columns={"dt_unix": "dt"})
        return ParsedResponse(r.to_dict(orient="records"), 51.17, 71.45, 0, 0, None)

    def forecast(self, lat, lon):
        self.n_requests += 1
        r = self.raw.tail(96).rename(columns={"dt_unix": "dt"}).copy()
        r["dt"] = r["dt"] + 96 * 3600
        return ParsedResponse(r.to_dict(orient="records"), 51.17, 71.45, 0, 0, None)


@pytest.fixture(scope="module")
def trained_project(tmp_path_factory):
    """Run preprocess + train on SYNTHETIC data in an isolated project root."""
    from pathlib import Path
    root = tmp_path_factory.mktemp("proj")
    src = Path(__file__).resolve().parents[1] / "config.yaml"
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["training"].update({"n_iter": 2, "n_iter_quick": 2, "cv_splits": 3, "n_jobs": 1,
                             "feature_configs": ["basic", "full_selected", "full_weather"],
                             "ablation_configs": ["full_weather_past"]})
    if not have_xgboost():
        data["training"]["models"] = ["ridge", "random_forest", "svr"]
    (root / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(root / "config.yaml")
    cfg.ensure_dirs()
    raw = synthetic_raw()
    raw.to_csv(cfg.path("raw_csv"), index=False)
    save_weather(synthetic_weather(), cfg.path("weather_csv"))
    run_preprocess(cfg)
    run_features(cfg)
    out = run_training(cfg, quick=True)
    return cfg, raw, out


def test_end_to_end_training_outputs(trained_project):
    cfg, _, out = trained_project
    comp = pd.read_csv(cfg.path("metrics_dir") / "model_comparison.csv")
    assert "persistence" in set(comp["model"])
    assert comp["test_n"].nunique() == 1  # identical eligible test dates for all approaches
    assert (cfg.path("models_dir") / "benchmark" / "best_model.joblib").exists()
    pt = pd.read_csv(cfg.path("predictions_dir") / "predictions_test.csv")
    assert {"issue_time_local", "target_date", "actual", "predicted", "abs_error", "model"} <= set(pt.columns)
    sel = json.loads((cfg.path("metrics_dir") / "model_selection.json").read_text())
    assert "validation" in sel["criterion"]


def test_evaluation_and_explainability_run(trained_project):
    from air_quality.evaluation import run_evaluation
    from air_quality.explainability import run_explainability
    cfg, _, _ = trained_project
    run_evaluation(cfg)
    run_explainability(cfg, n_repeats=2)
    assert (cfg.path("figures_dir") / "11_model_comparison.png").exists()
    assert (cfg.path("metrics_dir") / "E6_permutation_importance_test.csv").exists()


def test_successful_end_to_end_prediction(trained_project):
    cfg, raw, _ = trained_project
    now = pd.Timestamp("2025-12-20 19:30", tz="Asia/Almaty")
    r = make_forecast(cfg, client=FakeClient(raw), now=now, model="benchmark", with_provider=True, save=True,
                      weather_source=lambda c: synthetic_weather("2025-12-10", "2025-12-25"))
    assert r["prediction_date"] == "2025-12-21"
    assert r["label"] == "Our ML Forecast"
    assert np.isfinite(r["predicted_pm25"]) and r["units"] == "µg/m³"
    assert r["provider_forecast"]["label"] == "OpenWeatherMap Provider Forecast"
    assert (cfg.path("predictions_dir") / "latest_forecast.json").exists()


def test_prediction_before_issue_hour_strict_and_latest(trained_project):
    cfg, raw, _ = trained_project
    now = pd.Timestamp("2025-12-20 10:00", tz="Asia/Almaty")
    with pytest.raises(ForecastTimingError):
        make_forecast(cfg, client=FakeClient(raw), now=now, model="benchmark", save=False)
    r = make_forecast(cfg, client=FakeClient(raw), now=now, model="benchmark", issue_mode="latest", save=False)
    assert r["prediction_date"] == "2025-12-20" and "latest" in r["issue_protocol"]


def test_stale_data_detected(trained_project):
    cfg, raw, _ = trained_project
    old = raw[raw.dt_unix < int(pd.Timestamp("2025-12-19 00:00", tz="UTC").timestamp())]
    now = pd.Timestamp("2025-12-20 19:30", tz="Asia/Almaty")
    with pytest.raises(StaleDataError):
        make_forecast(cfg, client=FakeClient(old), now=now, model="benchmark", save=False)
    r = make_forecast(cfg, client=FakeClient(old), now=now, model="benchmark", force=True, save=False)
    assert r["reliable"] is False and r["warnings"]


def test_api_error_message(trained_project, monkeypatch, capsys):
    cfg, raw, _ = trained_project
    import air_quality.prediction as pred
    monkeypatch.setattr(pred.OpenWeatherAirPollutionClient, "from_config",
                        classmethod(lambda c, cfg_, key, **kw: FakeClient(raw, APIAuthError("HTTP 401 Unauthorized", 401))))
    monkeypatch.setenv("OPENWEATHER_API_KEY", "dummy_key_for_test_only")
    rc = cli.main(["--config", str(cfg.source), "predict", "--model", "benchmark", "--issue-mode", "latest"])
    assert rc == cli.EXIT_API
    assert "401" in capsys.readouterr().err


def test_missing_model_artifact(tmp_project, capsys, monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "dummy_key_for_test_only")
    rc = cli.main(["--config", str(tmp_project.source), "predict"])
    assert rc == cli.EXIT_MODEL
    assert "No trained model" in capsys.readouterr().err


def test_invalid_configuration(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("project: {}\n")
    rc = cli.main(["--config", str(tmp_path / "config.yaml"), "audit"])
    assert rc == cli.EXIT_CONFIG
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_missing_api_key_exit_code(tmp_project, monkeypatch, capsys):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.setattr("air_quality.config.find_project_root", lambda start=None: tmp_project.root)
    rc = cli.main(["--config", str(tmp_project.source), "test-api"])
    assert rc == cli.EXIT_KEY


def test_operational_model_separate(trained_project):
    from air_quality.training import train_operational
    cfg, _, _ = trained_project
    p = train_operational(cfg)
    b = joblib.load(p)
    assert "operational" in str(p) and "NOT the model evaluated" in b["note"]
    assert (cfg.path("models_dir") / "benchmark" / "best_model.joblib").exists()


def test_report_marks_missing_results_as_pending(tmp_project):
    import shutil
    from pathlib import Path
    from air_quality.report import build_report
    src = Path(__file__).resolve().parents[1] / "docs"
    shutil.copytree(src, tmp_project.root / "docs", dirs_exist_ok=True)
    text = build_report(tmp_project).read_text(encoding="utf-8")
    assert "PENDING" in text and "{{" not in text


def test_download_resume_split_and_manifest(tmp_project):
    """Chunks are cached, truncated chunks are split, and a rerun makes no new requests."""
    from air_quality.download import combine_chunks, download_history
    raw = synthetic_raw("2025-01-01", "2025-01-15", gaps=False, dups=False)

    class Client(FakeClient):
        def history(self, lat, lon, start, end):
            r = super().history(lat, lon, start, end)
            if end - start > 5 * 86400:  # simulate truncation of long requests
                r = ParsedResponse(r.records[:24], 51.17, 71.45, 0, 0, None)
            r.raw = {"coord": {"lat": 51.17, "lon": 71.45},
                     "list": [{"dt": x["dt"], "main": {"aqi": x["aqi"]},
                               "components": {k: x[k] for k in ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]}}
                              for x in r.records]}
            return r

    s = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp())
    e = int(pd.Timestamp("2025-01-15", tz="UTC").timestamp())
    c = Client(raw)
    summ = download_history(tmp_project, c, s, e, chunk_days=7)
    assert summ.n_chunks_split >= 1 and summ.n_chunks_failed == 0
    df, info = combine_chunks(tmp_project, s, e)
    assert len(df) == 14 * 24 and info["duplicates_removed"] == 0
    c2 = Client(raw)
    download_history(tmp_project, c2, s, e, chunk_days=7)
    assert c2.n_requests == 0  # everything served from cache


def test_ensemble_and_weather_configs_trained(trained_project):
    cfg, _, _ = trained_project
    res = pd.read_csv(cfg.path("metrics_dir") / "all_results.csv")
    assert "ensemble" in set(res["model"])
    assert {"full_weather", "full_weather_past"} <= set(res["feature_config"])
    assert (cfg.path("models_dir") / "benchmark" / "best_model_no_weather.joblib").exists()


def test_weather_fallback_when_forecast_unavailable(trained_project):
    from air_quality.weather import WeatherError
    cfg, raw, _ = trained_project

    def broken(c):
        raise WeatherError("offline")
    now = pd.Timestamp("2025-12-20 19:30", tz="Asia/Almaty")
    b = joblib.load(cfg.path("models_dir") / "benchmark" / "best_model.joblib")
    r = make_forecast(cfg, client=FakeClient(raw), now=now, model="benchmark", save=False, weather_source=broken)
    assert np.isfinite(r["predicted_pm25"])
    if any(f.startswith("wx") for f in b["features"]):
        assert "fallback" in r["weather_data"]
