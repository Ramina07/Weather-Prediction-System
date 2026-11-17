"""Regression metrics and the experiment matrix E1-E7 (tables + figures).

All numbers are computed from saved genuine model predictions
(``artifacts/predictions/*.csv``) produced by :mod:`air_quality.training`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, get_logger
from .models import DISPLAY_NAMES


# --------------------------------------------------------------------------- metrics
def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def smape(y, p, eps: float = 1e-6) -> float:
    """Symmetric MAPE in percent: 100/n * sum(2|y-p| / (|y|+|p|+eps))."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(100 * np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + eps)))


def mape(y, p, min_abs: float = 1.0) -> float:
    """MAPE in percent, computed only where |y| >= min_abs ug/m3 (avoids blow-up near 0)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.abs(y) >= min_abs
    return float(100 * np.mean(np.abs(y[m] - p[m]) / np.abs(y[m]))) if m.any() else float("nan")


def bias(y, p) -> float:
    """Mean prediction error (p - y); positive = over-forecast."""
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def all_metrics(y, p, eps: float = 1e-6) -> dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p), "smape_pct": smape(y, p, eps),
            "mape_pct_y_ge_1": mape(y, p), "bias": bias(y, p), "n": int(len(y))}


def skill_vs_persistence(mae_model: float, mae_persistence: float) -> float:
    """Skill score S = 1 - MAE_model / MAE_persistence (>0 means better than persistence)."""
    return float(1 - mae_model / mae_persistence) if mae_persistence > 0 else float("nan")


# --------------------------------------------------------------------------- helpers
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


UNIT = "µg/m³"
PALETTE = {"ridge": "#4C72B0", "random_forest": "#55A868", "xgboost": "#C44E52", "svr": "#8172B2", "ensemble": "#DD8452",
           "persistence": "#7F7F7F"}


def _label(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


# --------------------------------------------------------------------------- experiments
def run_evaluation(cfg: Config) -> dict[str, Any]:
    """Build experiment tables E1-E7 and figures 7-15 from saved predictions."""
    log = get_logger()
    mdir, fdir, pdir = cfg.path("metrics_dir"), cfg.path("figures_dir"), cfg.path("predictions_dir")
    res_path = mdir / "all_results.csv"
    if not res_path.exists():
        raise FileNotFoundError("artifacts/metrics/all_results.csv not found. Run the train command first.")
    res = pd.read_csv(res_path)
    sel = json.loads((mdir / "model_selection.json").read_text(encoding="utf-8"))
    pt = pd.read_csv(pdir / "predictions_test.csv", parse_dates=["target_date"])
    pv = pd.read_csv(pdir / "predictions_validation.csv", parse_dates=["target_date"])
    feats = pd.read_csv(cfg.path("features_csv"))
    plt = _plt()
    out: dict[str, Any] = {}
    best_model, best_cfg = sel["best"]["model"], sel["best"]["feature_config"]

    # ---------------- E1 model comparison (each model at its validation-selected config)
    e1 = pd.read_csv(mdir / "model_comparison.csv")
    out["E1"] = e1.to_dict(orient="records")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, met, lab in zip(axes, ["test_mae", "test_rmse", "test_r2"], [f"MAE ({UNIT})", f"RMSE ({UNIT})", "R²"]):
        colors = [PALETTE.get(m, "#999") for m in e1["model"]]
        ax.bar([_label(m) for m in e1["model"]], e1[met], color=colors)
        ax.set_title(f"2025 test {lab.split(' ')[0]}")
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(e1[met]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fig. 11. Model comparison on the independent 2025 test set (same eligible dates)")
    _save(fig, fdir / "11_model_comparison.png")

    # ---------------- E2 feature engineering (basic vs lag vs lag+rolling), tuned
    tuned = res[(res["variant"] == "tuned")]
    e2 = tuned[tuned["feature_config"].isin(["basic", "lag", "full"])].pivot_table(
        index=["model", "feature_config"], columns="stage", values=["mae", "rmse", "r2"]).round(4)
    e2.columns = [f"{s}_{m}" for m, s in e2.columns]
    e2 = e2.reset_index()
    e2.to_csv(mdir / "E2_feature_engineering.csv", index=False)
    out["E2"] = e2.to_dict(orient="records")

    # ---------------- E3 feature selection
    e3 = tuned[tuned["feature_config"].isin(["full", "full_selected"])][
        ["model", "feature_config", "stage", "mae", "rmse", "r2", "n_features_in", "n_features_used"]]
    e3.to_csv(mdir / "E3_feature_selection.csv", index=False)
    out["E3"] = e3.to_dict(orient="records")

    # ---------------- ablation chart (E2 + E3 + AQI) on validation
    abl = tuned[tuned["stage"] == "validation"].pivot_table(index="feature_config", columns="model", values="mae")
    order = [c for c in ["basic", "lag", "full", "full_selected", "full_aqi"] if c in abl.index]
    abl = abl.loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / max(len(abl.columns), 1)
    for i, m in enumerate(abl.columns):
        ax.bar(np.arange(len(abl)) + i * w, abl[m], w, label=_label(m), color=PALETTE.get(m))
    pv_pers = res[(res["model"] == "persistence") & (res["stage"] == "validation")]["mae"]
    if len(pv_pers):
        ax.axhline(pv_pers.iloc[0], color="k", ls="--", lw=1, label="Persistence")
    ax.set_xticks(np.arange(len(abl)) + w * (len(abl.columns) - 1) / 2)
    ax.set_xticklabels(abl.index)
    ax.set_ylabel(f"Validation MAE ({UNIT})")
    ax.set_xlabel("Feature configuration")
    ax.set_title("Fig. 13. Ablation: feature configuration vs 2024 validation MAE (tuned models)")
    lo = np.nanmin(abl.values)
    hi = max(np.nanmax(abl.values), pv_pers.iloc[0] if len(pv_pers) else 0)
    ax.set_ylim(lo * 0.9, hi * 1.05)  # zoomed axis; absolute values are in ablation_validation_mae.csv
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _save(fig, fdir / "13_ablation_feature_configs.png")
    abl.round(4).to_csv(mdir / "ablation_validation_mae.csv")

    # ---------------- E4 default vs tuned
    sub = res[res["model"] != "persistence"]
    e4 = sub.pivot_table(index=["model", "feature_config", "stage"], columns="variant", values="mae").reset_index()
    if {"default", "tuned"} <= set(e4.columns):
        e4["mae_change_tuned_minus_default"] = e4["tuned"] - e4["default"]
    e4.to_csv(mdir / "E4_hyperparameter_optimization.csv", index=False)
    out["E4"] = e4.round(4).to_dict(orient="records")

    # ---------------- per-model test predictions at selected configs
    chosen = {r["model"]: r["feature_config"] for r in sel["per_model"]}
    ptc = pt[(pt["variant"] == "tuned") & pt.apply(lambda r: chosen.get(r["model"]) == r["feature_config"], axis=1)]
    ptc = pd.concat([ptc, pt[pt["model"] == "persistence"]])
    best_pt = ptc[ptc["model"] == best_model].sort_values("target_date")
    pers_pt = ptc[ptc["model"] == "persistence"].sort_values("target_date")

    # ---------------- E5 elevated pollution
    train_y = feats[(feats["split"] == "train") & feats["eligible"]]["target_pm25"]
    thr_q = float(train_y.quantile(cfg["evaluation"]["elevated_quantile"]))
    who = float(cfg["evaluation"]["who_24h_guideline"])
    rows = []
    for m, g in ptc.groupby("model"):
        for name, thr in [(f"train_q{int(100 * cfg['evaluation']['elevated_quantile'])}", thr_q), ("WHO_24h_15", who)]:
            h = g[g["actual"] > thr]
            n = g[g["actual"] <= thr]
            rows.append({"model": m, "threshold_name": name, "threshold_ugm3": round(thr, 3), "n_elevated_days": len(h),
                         "mae_elevated": mae(h["actual"], h["predicted"]) if len(h) else np.nan,
                         "rmse_elevated": rmse(h["actual"], h["predicted"]) if len(h) else np.nan,
                         "bias_elevated": bias(h["actual"], h["predicted"]) if len(h) else np.nan,
                         "mae_non_elevated": mae(n["actual"], n["predicted"]) if len(n) else np.nan,
                         "n_non_elevated_days": len(n)})
    e5 = pd.DataFrame(rows)
    e5.to_csv(mdir / "E5_elevated_pollution.csv", index=False)
    out["E5"] = {"threshold_train_quantile": thr_q, "who_guideline": who, "table": e5.round(4).to_dict(orient="records")}

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(best_pt["target_date"], best_pt["actual"], color="k", lw=1, label="Observed (API) daily mean")
    ax.plot(best_pt["target_date"], best_pt["predicted"], color=PALETTE.get(best_model), lw=1,
            label=f"Our ML forecast: {_label(best_model)}")
    ax.plot(pers_pt["target_date"], pers_pt["predicted"], color="#7F7F7F", lw=0.8, ls=":", label="Persistence")
    ax.axhline(thr_q, color="orange", ls="--", lw=1, label=f"Elevated threshold (train q90) = {thr_q:.1f}")
    hi = best_pt[best_pt["actual"] > thr_q]
    ax.scatter(hi["target_date"], hi["actual"], color="orange", s=14, zorder=5, label=f"Elevated days (n={len(hi)})")
    ax.set_ylabel(f"Daily mean PM2.5 ({UNIT})")
    ax.set_xlabel("Target date (2025, local)")
    ax.set_title("Fig. 15. Predictions during elevated-pollution days (2025 independent test)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, fdir / "15_elevated_pollution_predictions.png")

    # ---------------- Fig 8-10 actual vs predicted, residuals
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(best_pt["actual"], best_pt["predicted"], s=12, alpha=0.6, color=PALETTE.get(best_model))
    lim = [0, max(best_pt["actual"].max(), best_pt["predicted"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="1:1 line")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Observed next-day mean PM2.5 ({UNIT})")
    ax.set_ylabel(f"Predicted next-day mean PM2.5 ({UNIT})")
    ax.set_title(f"Fig. 8. Actual vs predicted, {_label(best_model)} (2025 test)")
    ax.legend()
    _save(fig, fdir / "08_actual_vs_predicted.png")

    resid = best_pt["actual"] - best_pt["predicted"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(resid, bins=40, color=PALETTE.get(best_model), alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(f"Residual = observed − predicted ({UNIT})")
    ax.set_ylabel("Number of days")
    ax.set_title(f"Fig. 9. Residual distribution, {_label(best_model)} (2025 test)")
    _save(fig, fdir / "09_residual_distribution.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(best_pt["predicted"], resid, s=12, alpha=0.6, color=PALETTE.get(best_model))
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(f"Predicted PM2.5 ({UNIT})")
    ax.set_ylabel(f"Residual ({UNIT})")
    ax.set_title("Fig. 10. Residuals vs predicted values (2025 test)")
    _save(fig, fdir / "10_residuals_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(best_pt["target_date"], best_pt["actual"], color="k", lw=1, label="Observed (API)")
    ax.plot(best_pt["target_date"], best_pt["predicted"], color=PALETTE.get(best_model), lw=1, label=_label(best_model))
    ax.set_ylabel(f"Daily mean PM2.5 ({UNIT})")
    ax.set_xlabel("Target date")
    ax.set_title("Fig. 8b. Next-day forecast vs observation over the 2025 test year")
    ax.legend()
    _save(fig, fdir / "08b_test_timeseries.png")

    # ---------------- E7 time-based robustness
    ptc = ptc.copy()
    ptc["month"] = ptc["target_date"].dt.month
    ptc["season"] = ptc["month"].map(lambda m: ["Winter", "Spring", "Summer", "Autumn"][
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(m, 3)])
    hs = cfg["evaluation"]["heating_season"]
    md = ptc["target_date"].dt.strftime("%m-%d")
    ptc["heating"] = np.where((md >= hs["start"]) | (md <= hs["end"]), "Heating", "Non-heating")
    agg = lambda g: pd.Series({"n": len(g), "mae": mae(g["actual"], g["predicted"]),  # noqa: E731
                               "rmse": rmse(g["actual"], g["predicted"]), "bias": bias(g["actual"], g["predicted"])})
    e7m = ptc.groupby(["model", "month"]).apply(agg, include_groups=False).reset_index()
    e7s = ptc.groupby(["model", "season"]).apply(agg, include_groups=False).reset_index()
    e7h = ptc.groupby(["model", "heating"]).apply(agg, include_groups=False).reset_index()
    e7m.to_csv(mdir / "E7_monthly_errors.csv", index=False)
    e7s.to_csv(mdir / "E7_seasonal_errors.csv", index=False)
    e7h.to_csv(mdir / "E7_heating_errors.csv", index=False)
    out["E7"] = {"seasonal": e7s.round(4).to_dict(orient="records"), "heating": e7h.round(4).to_dict(orient="records")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, met in zip(axes, ["mae", "rmse"]):
        for m, g in e7m.groupby("model"):
            ax.plot(g["month"], g[met], marker="o", label=_label(m), color=PALETTE.get(m),
                    ls="--" if m == "persistence" else "-")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Target month (2025)")
        ax.set_ylabel(f"{met.upper()} ({UNIT})")
        ax.set_title(f"Monthly {met.upper()}")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig. 14. Monthly forecasting error on the 2025 test set")
    _save(fig, fdir / "14_monthly_error.png")

    # ---------------- training/validation summary
    out["selection"] = sel
    (mdir / "evaluation_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("Evaluation tables and figures written to %s and %s", mdir, fdir)
    return out
