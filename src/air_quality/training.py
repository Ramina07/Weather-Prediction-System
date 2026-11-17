"""Training, chronological validation, hyperparameter search and benchmark
freezing for the four models and the persistence baseline.

Protocol (per model m and feature configuration c):

1. RandomizedSearchCV on the TRAIN period (targets 2021-2023) with
   ``TimeSeriesSplit(n_splits, gap)``; scoring = negative MAE. All
   preprocessing is inside the pipeline, so it is refitted on each CV fold.
2. The best configuration (refitted on all TRAIN samples) is evaluated on the
   VALIDATION year (2024). A default-hyperparameter pipeline is evaluated the
   same way (Experiment E4).
3. Model selection uses validation MAE only.
4. Each configuration is refitted on TRAIN+VALIDATION with the hyperparameters
   chosen in step 1 and evaluated ONCE on the TEST year (2025). Test results
   are frozen and never used to change any hyperparameter.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from . import __version__
from .config import Config, get_logger
from .evaluation import all_metrics, skill_vs_persistence
from .feature_engineering import (FEATURE_VERSION, PERSISTENCE_FEATURE, WEATHER_CONFIGS, WEATHER_FEATURES,
                                  features_for_config)
from .feature_selection import selected_feature_names
from .models import ENSEMBLE_MEMBERS, PersistenceForecaster, build_pipeline, make_ensemble, param_space

TARGET = "target_pm25"


def load_model_frame(cfg: Config, path: Path | None = None) -> pd.DataFrame:
    path = path or cfg.path("features_csv")
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}. Run the preprocess command first.")
    df = pd.read_csv(path)
    df = df[df["eligible"].astype(bool)].copy()
    df = df.sort_values("target_date").reset_index(drop=True)
    return df


def _json_params(p: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in p.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v if v is None or isinstance(v, (str, bool)) else str(v)
    return out


def _period(df: pd.DataFrame) -> str:
    return f"{df['target_date'].min()}..{df['target_date'].max()}" if len(df) else ""


def _pred_frame(df: pd.DataFrame, pred: np.ndarray, model: str, cfg_name: str, variant: str, stage: str) -> pd.DataFrame:
    return pd.DataFrame({"issue_time_local": df["issue_time_local"].values, "target_date": df["target_date"].values,
                         "actual": df[TARGET].values, "predicted": pred,
                         "abs_error": np.abs(df[TARGET].values - pred), "model": model,
                         "feature_config": cfg_name, "variant": variant, "stage": stage})


def make_cv(cfg: Config, n_samples: int) -> TimeSeriesSplit:
    t = cfg["training"]
    n_splits = min(t["cv_splits"], max(2, n_samples // 60))
    return TimeSeriesSplit(n_splits=n_splits, gap=int(t["cv_gap_days"]))


def run_training(cfg: Config, quick: bool = False, models: list[str] | None = None,
                 configs: list[str] | None = None) -> dict[str, Any]:
    log = get_logger()
    t = cfg["training"]
    seed, n_jobs = cfg.seed, t["n_jobs"]
    n_iter = t["n_iter_quick"] if quick else t["n_iter"]
    models = models or t["models"]
    main_cfgs = configs or t["feature_configs"]
    abl_cfgs = [] if configs else t.get("ablation_configs", [])
    df = load_model_frame(cfg)
    tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
    trva = pd.concat([tr, va])
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"Empty split: train={len(tr)}, validation={len(va)}, test={len(te)}")
    log.info("Eligible samples: train=%d (%s), validation=%d (%s), test=%d (%s)",
             len(tr), _period(tr), len(va), _period(va), len(te), _period(te))
    periods = {"train": _period(tr), "validation": _period(va), "test": _period(te)}
    mdir, pdir = cfg.path("metrics_dir"), cfg.path("predictions_dir")
    bdir = cfg.path("models_dir") / "benchmark"
    for d in (mdir, pdir, bdir):
        d.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    preds_val, preds_test, cv_tables = [], [], []
    fitted: dict[tuple[str, str], dict[str, Any]] = {}

    # ---------------------------------------------------------------- persistence
    pers = PersistenceForecaster(PERSISTENCE_FEATURE).fit(None)
    for stage, part in (("validation", va), ("test", te)):
        t0 = time.perf_counter()
        p = pers.predict(part)
        dt = time.perf_counter() - t0
        m = all_metrics(part[TARGET], p, cfg["evaluation"]["smape_epsilon"])
        results.append({"model": "persistence", "feature_config": PERSISTENCE_FEATURE, "variant": "baseline",
                        "stage": stage, **m, "train_seconds": 0.0, "predict_seconds": dt, "cv_mae": np.nan,
                        "best_params": "{}", "n_features_in": 1, "n_features_used": 1, **_periods(periods)})
        (preds_val if stage == "validation" else preds_test).append(
            _pred_frame(part, p, "persistence", PERSISTENCE_FEATURE, "baseline", stage))
    pers_mae = {r["stage"]: r["mae"] for r in results}

    # ---------------------------------------------------------------- ML models
    eps = cfg["evaluation"]["smape_epsilon"]

    def evaluate(name: str, cfg_name: str, feats: list[str], variants: dict[str, Any], best_params: dict,
                 cv_mae: float, tune_s: float, dflt_s: float) -> None:
        """Validation (train-only fit) + single test (train+validation refit) for each variant."""
        resid_val = None
        for variant, est in variants.items():
            t0 = time.perf_counter()
            pv = est.predict(va[feats])
            pdt = time.perf_counter() - t0
            m = all_metrics(va[TARGET], pv, eps)
            used = selected_feature_names(est, feats)
            results.append({"model": name, "feature_config": cfg_name, "variant": variant, "stage": "validation",
                            **m, "skill_vs_persistence": skill_vs_persistence(m["mae"], pers_mae["validation"]),
                            "train_seconds": tune_s if variant == "tuned" else dflt_s, "predict_seconds": pdt,
                            "cv_mae": cv_mae if variant == "tuned" else np.nan,
                            "best_params": json.dumps(best_params if variant == "tuned" else {}),
                            "n_features_in": len(feats), "n_features_used": len(used), **_periods(periods)})
            preds_val.append(_pred_frame(va, pv, name, cfg_name, variant, "validation"))
            val_mae = m["mae"]
            if variant == "tuned":
                resid_val = va[TARGET].to_numpy() - pv
            # refit on train+validation with the SAME hyperparameters, test once
            final = clone(est)
            t0 = time.perf_counter()
            final.fit(trva[feats], trva[TARGET].to_numpy())
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pt = final.predict(te[feats])
            pdt = time.perf_counter() - t0
            m = all_metrics(te[TARGET], pt, eps)
            used = selected_feature_names(final, feats)
            results.append({"model": name, "feature_config": cfg_name, "variant": variant, "stage": "test",
                            **m, "skill_vs_persistence": skill_vs_persistence(m["mae"], pers_mae["test"]),
                            "train_seconds": fit_s, "predict_seconds": pdt, "cv_mae": np.nan,
                            "best_params": json.dumps(best_params if variant == "tuned" else {}),
                            "n_features_in": len(feats), "n_features_used": len(used), **_periods(periods)})
            preds_test.append(_pred_frame(te, pt, name, cfg_name, variant, "test"))
            if variant == "tuned":
                fitted[(name, cfg_name)] = {"pipeline": final, "features": feats, "selected": used,
                                            "params": best_params, "resid_val": resid_val, "val_mae": val_mae}
                log.info("[%s | %s] CV MAE=%.3f | val MAE tuned=%.3f | test MAE tuned=%.3f",
                         name, cfg_name, cv_mae, val_mae, m["mae"])

    has_weather = tr[WEATHER_FEATURES].notna().any().any()
    for cfg_name in main_cfgs + abl_cfgs:
        if cfg_name in WEATHER_CONFIGS and not has_weather:
            log.warning("Skipping %s: no weather data (run the `weather` command first)", cfg_name)
            continue
        feats = features_for_config(cfg_name)
        select = cfg_name.endswith("_selected")
        tuned_est: dict[str, Any] = {}
        default_est: dict[str, Any] = {}
        for name in models:
            log.info("[%s | %s] tuning (n_iter=%d) ...", name, cfg_name, n_iter)
            Xtr, ytr = tr[feats], tr[TARGET].to_numpy()
            pipe = build_pipeline(name, select, seed, n_jobs)
            inner_parallel = name in ("random_forest", "xgboost")
            search = RandomizedSearchCV(pipe, param_space(name, select), n_iter=n_iter,
                                        scoring="neg_mean_absolute_error", cv=make_cv(cfg, len(tr)),
                                        random_state=seed, n_jobs=1 if inner_parallel else n_jobs,
                                        refit=True, error_score="raise")
            t0 = time.perf_counter()
            search.fit(Xtr, ytr)
            tune_s = time.perf_counter() - t0
            cvr = pd.DataFrame(search.cv_results_)
            cvr.insert(0, "model", name)
            cvr.insert(1, "feature_config", cfg_name)
            cv_tables.append(cvr[["model", "feature_config", "params", "mean_test_score", "std_test_score",
                                  "rank_test_score", "mean_fit_time"]])
            dflt = build_pipeline(name, select, seed, n_jobs)
            t0 = time.perf_counter()
            dflt.fit(Xtr, ytr)
            dflt_s = time.perf_counter() - t0
            tuned_est[name], default_est[name] = search.best_estimator_, dflt
            evaluate(name, cfg_name, feats, {"tuned": search.best_estimator_, "default": dflt},
                     _json_params(search.best_params_), -search.best_score_, tune_s, dflt_s)

        # MODEL 5: equal-weight ensemble of the tuned RF, XGBoost and SVR (no extra tuning)
        members = [m for m in ENSEMBLE_MEMBERS if m in tuned_est]
        if t.get("ensemble", True) and len(members) >= 2:
            t0 = time.perf_counter()
            ens = make_ensemble({m: tuned_est[m] for m in members}).fit(tr[feats], tr[TARGET].to_numpy())
            ens_s = time.perf_counter() - t0
            ens_d = make_ensemble({m: default_est[m] for m in members}).fit(tr[feats], tr[TARGET].to_numpy())
            evaluate("ensemble", cfg_name, feats, {"tuned": ens, "default": ens_d},
                     {"members": ",".join(members), "weights": "equal"}, np.nan, ens_s, ens_s)

    res = pd.DataFrame(results)
    res.to_csv(mdir / "all_results.csv", index=False)
    pd.concat(preds_val).to_csv(pdir / "predictions_validation.csv", index=False)
    pd.concat(preds_test).to_csv(pdir / "predictions_test.csv", index=False)
    pd.concat(cv_tables).to_csv(mdir / "cv_results.csv", index=False)

    # ---------------------------------------------------------------- selection (validation only)
    val_tuned = res[(res["stage"] == "validation") & (res["variant"] == "tuned") & res["feature_config"].isin(main_cfgs)]
    per_model = val_tuned.loc[val_tuned.groupby("model")["mae"].idxmin(), ["model", "feature_config", "mae"]]
    best_row = val_tuned.loc[val_tuned["mae"].idxmin()]
    selection = {"criterion": "lowest 2024 validation MAE among tuned models and main feature configurations",
                 "best": {"model": best_row["model"], "feature_config": best_row["feature_config"],
                          "validation_mae": float(best_row["mae"])},
                 "per_model": per_model.rename(columns={"mae": "validation_mae"}).to_dict(orient="records"),
                 "periods": periods, "n_samples": {"train": len(tr), "validation": len(va), "test": len(te)},
                 "persistence_mae": pers_mae}
    (mdir / "model_selection.json").write_text(json.dumps(selection, indent=2, default=float), encoding="utf-8")

    # ---------------------------------------------------------------- E1 comparison table
    rows = []
    for _, r in per_model.iterrows():
        v = res[(res.model == r.model) & (res.feature_config == r.feature_config) & (res.variant == "tuned")]
        rows.append(_comparison_row(v, r.model, r.feature_config, periods))
    rows.append(_comparison_row(res[res.model == "persistence"], "persistence", PERSISTENCE_FEATURE, periods))
    comp = pd.DataFrame(rows)
    comp["test_skill_vs_persistence"] = 1 - comp["test_mae"] / pers_mae["test"]
    comp["test_mae_improvement_pct"] = 100 * comp["test_skill_vs_persistence"]
    comp = comp.sort_values("validation_mae").reset_index(drop=True)
    comp.to_csv(mdir / "model_comparison.csv", index=False)
    (mdir / "model_comparison.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- save frozen benchmark models
    q = cfg["evaluation"]["interval_coverage"]
    for (name, cfg_name), info in fitted.items():
        if per_model.set_index("model").loc[name, "feature_config"] != cfg_name:
            continue
        test_row = comp[comp.model == name].iloc[0].to_dict()
        bundle = _bundle(cfg, name, cfg_name, info, periods, "train+validation", test_row, q)
        joblib.dump(bundle, bdir / f"{name}.joblib")
        if name == best_row["model"] and cfg_name == best_row["feature_config"]:
            joblib.dump(bundle, bdir / "best_model.joblib")
            card = {k: v for k, v in bundle.items() if k != "pipeline"}
            (bdir / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    # fallback without weather features (used live if the weather forecast is unavailable)
    nw = val_tuned[~val_tuned["feature_config"].isin(WEATHER_CONFIGS)]
    if len(nw):
        r = nw.loc[nw["mae"].idxmin()]
        info = fitted[(r["model"], r["feature_config"])]
        test_r = res[(res.model == r["model"]) & (res.feature_config == r["feature_config"]) & (res.variant == "tuned")
                     & (res.stage == "test")].iloc[0].to_dict()
        joblib.dump(_bundle(cfg, r["model"], r["feature_config"], info, periods, "train+validation", test_r, q),
                    bdir / "best_model_no_weather.joblib")
        selection["best_without_weather"] = {"model": r["model"], "feature_config": r["feature_config"],
                                             "validation_mae": float(r["mae"])}
        (mdir / "model_selection.json").write_text(json.dumps(selection, indent=2, default=float), encoding="utf-8")
    log.info("Best model on validation: %s (%s), validation MAE = %.3f",
             best_row["model"], best_row["feature_config"], best_row["mae"])
    return {"results": res, "comparison": comp, "selection": selection}


def _periods(p: dict[str, str]) -> dict[str, str]:
    return {"train_period": p["train"], "validation_period": p["validation"], "test_period": p["test"]}


def _comparison_row(v: pd.DataFrame, model: str, cfg_name: str, periods: dict[str, str]) -> dict[str, Any]:
    va, te = v[v.stage == "validation"].iloc[0], v[v.stage == "test"].iloc[0]
    return {"model": model, "feature_config": cfg_name, **_periods(periods),
            "cv_mae": va.get("cv_mae", np.nan), "validation_mae": va["mae"], "validation_rmse": va["rmse"],
            "validation_r2": va["r2"], "test_mae": te["mae"], "test_rmse": te["rmse"], "test_r2": te["r2"],
            "test_smape_pct": te["smape_pct"], "test_mape_pct_y_ge_1": te["mape_pct_y_ge_1"], "test_bias": te["bias"],
            "test_n": int(te["n"]), "tuning_seconds": va["train_seconds"], "refit_seconds": te["train_seconds"],
            "test_predict_seconds": te["predict_seconds"], "n_features_used": int(te["n_features_used"]),
            "best_hyperparameters": va["best_params"]}


def _bundle(cfg: Config, name: str, cfg_name: str, info: dict[str, Any], periods: dict[str, str], fitted_on: str,
            test_row: dict[str, Any] | None, q: float) -> dict[str, Any]:
    lo, hi = np.quantile(info["resid_val"], [(1 - q) / 2, 1 - (1 - q) / 2]) if info.get("resid_val") is not None else (np.nan, np.nan)
    return {"pipeline": info["pipeline"], "model_name": name, "feature_config": cfg_name,
            "features": info["features"], "selected_features": info["selected"], "hyperparameters": info["params"],
            "fitted_on": fitted_on, "periods": periods, "target": "mean PM2.5 (ug/m3) of local day d+1",
            "forecast_protocol": {k: v for k, v in cfg["forecast"].items()},
            "location": cfg["location"], "feature_version": FEATURE_VERSION, "package_version": __version__,
            "residual_interval": {"coverage": q, "lower": float(lo), "upper": float(hi),
                                  "source": "2024 validation residuals of the train-only fit (empirical, approximate)"},
            "benchmark_test_metrics": test_row, "created_utc": datetime.now(timezone.utc).isoformat()}


def train_operational(cfg: Config) -> Path:
    """Refit the frozen best configuration on ALL labelled data (incl. 2026).

    Stored separately in artifacts/models/operational; the 2025 benchmark
    results always refer to the benchmark model, never to this one.
    """
    log = get_logger()
    bdir = cfg.path("models_dir") / "benchmark"
    best = bdir / "best_model.joblib"
    if not best.exists():
        raise FileNotFoundError("Benchmark model not found. Run the train command first.")
    b = joblib.load(best)
    src = cfg.path("features_operational_csv")
    df = load_model_frame(cfg, src if src.exists() else cfg.path("features_csv"))
    pipe = clone(b["pipeline"])
    pipe.fit(df[b["features"]], df[TARGET].to_numpy())
    info = {"pipeline": pipe, "features": b["features"], "selected": selected_feature_names(pipe, b["features"]),
            "params": b["hyperparameters"], "resid_val": None}
    periods = {"train": _period(df), "validation": "", "test": ""}
    bundle = _bundle(cfg, b["model_name"], b["feature_config"], info, periods, "all labelled data", None,
                     cfg["evaluation"]["interval_coverage"])
    bundle["residual_interval"] = b["residual_interval"]  # inherited from benchmark validation residuals
    bundle["note"] = ("Operational refit of the benchmark configuration; NOT the model evaluated on the 2025 test.")
    odir = cfg.path("models_dir") / "operational"
    odir.mkdir(parents=True, exist_ok=True)
    out = odir / "operational_model.joblib"
    joblib.dump(bundle, out)
    log.info("Operational model (%s, %s) refitted on %d samples (%s) -> %s",
             b["model_name"], b["feature_config"], len(df), periods["train"], out)
    return out
