"""Generate docs/research_report_draft.md from the template and real artifacts.

Every number in the generated report is read from files produced by the
pipeline. If an artifact is missing, the corresponding placeholder is filled
with an explicit "PENDING" marker — no value is ever invented.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, get_logger
from .models import DISPLAY_NAMES

PENDING = "**[PENDING — run `python -m air_quality.cli run-all` to generate this result]**"
HIST_PREFIXES = ("pm25_",)


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def _name(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


def _md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values]
    return "\n".join([head, sep, *rows])


def load_results(cfg: Config) -> dict[str, Any]:
    m = cfg.path("metrics_dir")
    return {
        "audit": _read_json(m / "dataset_audit.json"),
        "quality": _read_json(cfg.path("quality_report_json")),
        "meta": _read_json(cfg.path("raw_metadata")),
        "eda": _read_json(m / "eda_summary.json"),
        "selection": _read_json(m / "model_selection.json"),
        "comparison": _read_csv(m / "model_comparison.csv"),
        "all": _read_csv(m / "all_results.csv"),
        "e3": _read_csv(m / "E3_feature_selection.csv"),
        "e4": _read_csv(m / "E4_hyperparameter_optimization.csv"),
        "e5": _read_csv(m / "E5_elevated_pollution.csv"),
        "e6": _read_json(m / "E6_summary.json"),
        "e7s": _read_csv(m / "E7_seasonal_errors.csv"),
        "e7h": _read_csv(m / "E7_heating_errors.csv"),
        "env": _read_json(cfg.path("logs_dir") / "environment.json"),
        "forecast": _read_json(cfg.path("predictions_dir") / "latest_forecast.json"),
    }


def hypothesis_verdicts(cfg: Config, R: dict[str, Any] | None = None) -> dict[str, str]:
    """Evaluate H1-H5 from the actual results (never assumed)."""
    R = R or load_results(cfg)
    sel, comp, res = R["selection"], R["comparison"], R["all"]
    out: dict[str, str] = {}
    if sel is None or comp is None or res is None:
        return {k: PENDING for k in ["H1", "H2", "H3", "H4", "H5"]}
    best = sel["best"]["model"]
    out["H1"] = (f"{'Supported' if best in ('random_forest', 'xgboost') else 'Not supported'}: the lowest 2024 validation "
                 f"MAE was obtained by {_name(best)} ({sel['best']['feature_config']}, "
                 f"{sel['best']['validation_mae']:.3f} µg/m³).")
    v = res[(res.stage == "validation") & (res.variant == "tuned")].pivot_table(index="model", columns="feature_config", values="mae")
    if {"basic", "lag", "full"} <= set(v.columns):
        better = [(m, r["basic"], min(r["lag"], r["full"])) for m, r in v.iterrows()]
        n_imp = sum(b > e for _, b, e in better)
        detail = "; ".join(f"{_name(m)} {b:.3f}→{e:.3f}" for m, b, e in better)
        out["H2"] = (f"{'Supported' if n_imp >= 3 else ('Partly supported' if n_imp >= 1 else 'Not supported')}: "
                     f"engineered features (best of lag/full) lowered validation MAE for {n_imp} of {len(better)} models "
                     f"(basic → engineered: {detail}).")
    else:
        out["H2"] = PENDING
    b = comp[comp.model == best].iloc[0]
    p = comp[comp.model == "persistence"].iloc[0]
    n_beat = int((comp[comp.model != "persistence"]["test_mae"] < p["test_mae"]).sum())
    out["H3"] = (f"{'Supported' if b['test_mae'] < p['test_mae'] else 'Not supported'}: on the 2025 test the selected "
                 f"{_name(best)} obtained MAE {b['test_mae']:.3f} µg/m³ vs {p['test_mae']:.3f} µg/m³ for persistence "
                 f"(skill {b['test_skill_vs_persistence']:+.3f}, i.e. {b['test_mae_improvement_pct']:+.1f}% MAE change); "
                 f"{n_beat} of {len(comp) - 1} ML models beat persistence.")
    e6 = R["e6"]
    if e6 and e6.get("top_permutation_features_best_model"):
        top = [t["feature"] for t in e6["top_permutation_features_best_model"][:3]]
        hist = sum(f.startswith(HIST_PREFIXES) for f in top)
        out["H4"] = (f"{'Supported' if hist >= 2 else 'Not supported'}: the three most important features of "
                     f"{_name(best)} by test-set permutation importance were {', '.join(top)} "
                     f"({hist} of 3 are PM2.5-history features). Importance is associational, not causal.")
    else:
        out["H4"] = PENDING
    e5 = R["e5"]
    if e5 is not None:
        r = e5[(e5.model == best) & e5.threshold_name.str.startswith("train")]
        if len(r) and r.iloc[0]["n_elevated_days"] > 0:
            r = r.iloc[0]
            ratio = r["mae_elevated"] / r["mae_non_elevated"] if r["mae_non_elevated"] else float("nan")
            sup = ratio > 1.5 and r["bias_elevated"] < 0
            out["H5"] = (f"{'Supported' if sup else 'Not supported'}: on {int(r['n_elevated_days'])} elevated test days "
                         f"(> {r['threshold_ugm3']:.1f} µg/m³, training 90th percentile) the MAE of {_name(best)} was "
                         f"{r['mae_elevated']:.3f} vs {r['mae_non_elevated']:.3f} µg/m³ on other days (ratio {ratio:.2f}); "
                         f"bias on elevated days {r['bias_elevated']:+.3f} µg/m³."
                         + (" The elevated-day sample is small; interpret with caution." if r["n_elevated_days"] < 30 else ""))
        else:
            out["H5"] = "Not testable: no elevated days in the test period."
    else:
        out["H5"] = PENDING
    return out


def build_tokens(cfg: Config) -> dict[str, str]:
    R = load_results(cfg)
    T: dict[str, str] = {}
    a, q, eda, sel, comp = R["audit"], R["quality"], R["eda"], R["selection"], R["comparison"]
    T["N_RECORDS"] = f"{a['n_records']:,}" if a else PENDING
    T["PERIOD"] = f"{a['first_timestamp_local'][:16]} to {a['last_timestamp_local'][:16]} (local)" if a else PENDING
    T["COVERAGE"] = f"{a['coverage_pct']:.2f}%" if a else PENDING
    T["MISSING_HOURS"] = f"{a['missing_hours']:,}" if a else PENDING
    T["DUPLICATES"] = str(a["n_duplicated_timestamps"]) if a else PENDING
    T["VALID_DAYS"] = f"{a['n_valid_days_min_20h']:,} of {a['n_calendar_days_local']:,}" if a else PENDING
    T["PM25_RANGE"] = f"{a['pm2_5_min']:.2f}–{a['pm2_5_max']:.2f} µg/m³" if a else PENDING
    T["N_REQUESTS"] = str((R["meta"] or {}).get("download", {}).get("n_requests", "PENDING")) if R["meta"] else PENDING
    if q:
        T["QUALITY_TABLE"] = _md_table(pd.DataFrame({"before cleaning": q["before"], "after cleaning": q["after"]}).loc[
            ["n_records", "n_unique_timestamps", "n_duplicated_timestamps", "expected_hourly_intervals", "coverage_pct",
             "missing_hours", "pm2_5_min", "pm2_5_max", "n_valid_days_min_20h"]].reset_index().rename(columns={"index": "metric"}))
        T["MISSING_TABLE"] = _md_table(pd.DataFrame({"missing (observed hours)": q["after"]["missing_by_field"],
                                                     "%": q["after"]["missing_pct_by_field"]}).reset_index().rename(columns={"index": "field"}))
        T["GAPS"] = f"{q['gaps']['n_gaps']} gaps, longest {q['gaps']['max_gap_hours']} h, {q['gaps']['gaps_over_24h']} longer than 24 h"
    else:
        T["QUALITY_TABLE"] = T["MISSING_TABLE"] = T["GAPS"] = PENDING
    if eda:
        sm = eda["seasonal_mean"]
        hm = eda["heating_mean"]["mean"]
        T["EDA_SEASONAL"] = ", ".join(f"{k} {sm[k]:.2f}" for k in ["Winter", "Spring", "Summer", "Autumn"] if k in sm) + " µg/m³"
        T["EDA_HEATING"] = (f"{hm.get('Heating', float('nan')):.2f} µg/m³ in the heating season vs "
                            f"{hm.get('Non-heating', float('nan')):.2f} µg/m³ outside it")
        T["EDA_ACF"] = f"lag-1 daily autocorrelation {eda['acf_daily']['1']:.3f}, lag-7 {eda['acf_daily']['7']:.3f}"
        T["EDA_WHO"] = f"{eda['extremes']['daily_days_above_who_15']} of {eda['extremes']['n_daily_targets']} valid days"
        ss = eda["split_shift"]
        T["SPLIT_TABLE"] = _md_table(pd.DataFrame(ss).reset_index().rename(columns={"index": "split"})[
            ["split", "count", "mean", "std", "50%", "max"]], 2) if isinstance(ss, dict) else PENDING
    else:
        for k in ["EDA_SEASONAL", "EDA_HEATING", "EDA_ACF", "EDA_WHO", "SPLIT_TABLE"]:
            T[k] = PENDING
    if sel and comp is not None:
        n = sel["n_samples"]
        T["N_SPLITS"] = f"train {n['train']}, validation {n['validation']}, test {n['test']}"
        T["BEST_MODEL"] = _name(sel["best"]["model"])
        T["BEST_CFG"] = sel["best"]["feature_config"]
        b = comp[comp.model == sel["best"]["model"]].iloc[0]
        p = comp[comp.model == "persistence"].iloc[0]
        T["BEST_TEST_MAE"] = f"{b['test_mae']:.2f}"
        T["BEST_TEST_RMSE"] = f"{b['test_rmse']:.2f}"
        T["BEST_TEST_R2"] = f"{b['test_r2']:.3f}"
        T["PERS_TEST_MAE"] = f"{p['test_mae']:.2f}"
        T["PERS_TEST_RMSE"] = f"{p['test_rmse']:.2f}"
        T["PERS_TEST_R2"] = f"{p['test_r2']:.3f}"
        T["BEST_IMPROV"] = f"{b['test_mae_improvement_pct']:+.1f}%"
        show = comp[["model", "feature_config", "cv_mae", "validation_mae", "test_mae", "test_rmse", "test_r2",
                     "test_smape_pct", "test_bias", "test_mae_improvement_pct", "test_n"]].copy()
        show["model"] = show["model"].map(_name)
        T["TABLE_E1"] = _md_table(show)
        hp = comp[comp.model != "persistence"][["model", "feature_config", "best_hyperparameters", "tuning_seconds"]].copy()
        hp["model"] = hp["model"].map(_name)
        T["TABLE_HP"] = _md_table(hp, 1)
    else:
        for k in ["N_SPLITS", "BEST_MODEL", "BEST_CFG", "BEST_TEST_MAE", "BEST_TEST_RMSE", "BEST_TEST_R2", "PERS_TEST_MAE",
                  "PERS_TEST_RMSE", "PERS_TEST_R2", "BEST_IMPROV", "TABLE_E1", "TABLE_HP"]:
            T[k] = PENDING
    res = R["all"]
    if res is not None:
        v = res[(res.variant == "tuned")].pivot_table(index="feature_config", columns=["model", "stage"], values="mae")
        v.columns = [f"{_name(m)} {s[:4]}." for m, s in v.columns]
        order = [c for c in ["basic", "lag", "full", "full_selected", "full_aqi"] if c in v.index]
        T["TABLE_E2"] = _md_table(v.loc[order].reset_index())
    else:
        T["TABLE_E2"] = PENDING
    T["TABLE_E3"] = _md_table(R["e3"].assign(model=R["e3"]["model"].map(_name))) if R["e3"] is not None else PENDING
    if R["e4"] is not None:
        e4 = R["e4"][R["e4"]["stage"] == "validation"].copy()
        e4["model"] = e4["model"].map(_name)
        T["TABLE_E4"] = _md_table(e4)
    else:
        T["TABLE_E4"] = PENDING
    if R["e5"] is not None:
        e5 = R["e5"].copy()
        e5["model"] = e5["model"].map(_name)
        T["TABLE_E5"] = _md_table(e5)
    else:
        T["TABLE_E5"] = PENDING
    if R["e6"]:
        T["E6_TOP"] = ", ".join(f"{t['feature']} ({t['mae_increase_mean']:.3f})" for t in R["e6"]["top_permutation_features_best_model"][:8])
        T["E6_SHAP"] = R["e6"].get("shap", "skipped")
    else:
        T["E6_TOP"] = T["E6_SHAP"] = PENDING
    if R["e7s"] is not None:
        s = R["e7s"].pivot(index="season", columns="model", values="mae")
        s.columns = [_name(c) for c in s.columns]
        T["TABLE_E7S"] = _md_table(s.reset_index(), 2)
        h = R["e7h"].pivot(index="heating", columns="model", values="mae")
        h.columns = [_name(c) for c in h.columns]
        T["TABLE_E7H"] = _md_table(h.reset_index(), 2)
    else:
        T["TABLE_E7S"] = T["TABLE_E7H"] = PENDING
    for k, vtxt in hypothesis_verdicts(cfg, R).items():
        T[k] = vtxt
    f = R["forecast"]
    T["FORECAST"] = (f"Forecast issued {f['forecast_issued_local']} for {f['prediction_date']}: **{f['predicted_pm25']:.2f} µg/m³** "
                     f"({f['model']}, data freshness {f['data_freshness_latest_observation_local']}, reliable={f['reliable']})"
                     if f else PENDING)
    env = R["env"]
    T["ENV"] = (", ".join(f"{k} {v}" for k, v in env.items() if k != "platform") + f"; platform: {env.get('platform')}") if env else PENDING
    root = cfg.root
    T["LIT_REVIEW"] = _strip_h1((root / "docs" / "literature_review.md").read_text(encoding="utf-8"))
    T["REFERENCES"] = _strip_h1((root / "docs" / "references_apa.md").read_text(encoding="utf-8"))
    disc = root / "docs" / "results_discussion.md"
    T["DISCUSSION"] = _strip_h1(disc.read_text(encoding="utf-8")) if disc.exists() else PENDING
    T["MATH"] = _strip_h1((root / "docs" / "mathematical_formulations.md").read_text(encoding="utf-8"))
    return T


def _strip_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def build_report(cfg: Config) -> Path:
    log = get_logger()
    tpl_path = cfg.root / "docs" / "templates" / "research_report_template.md"
    tpl = tpl_path.read_text(encoding="utf-8")
    tokens = build_tokens(cfg)
    out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: tokens.get(m.group(1), f"{{{{{m.group(1)}}}}}"), tpl)
    p = cfg.root / "docs" / "research_report_draft.md"
    p.write_text(out, encoding="utf-8")
    n_pending = out.count("[PENDING")
    log.info("Report written to %s (%d pending placeholders)", p, n_pending)
    return p
