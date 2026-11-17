"""Generate the five project notebooks as nbformat-v4 JSON (no execution here)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True, parents=True)

SETUP = '''# --- Setup: locate the project root and import the package --------------------
import os, sys, json
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
from air_quality.config import load_config, setup_logging
cfg = load_config(ROOT / "config.yaml"); setup_logging(cfg, "notebooks")
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
FIG = cfg.path("figures_dir"); MET = cfg.path("metrics_dir")
print("Project root:", ROOT)'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def write(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:03d}"
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


GUARD = '''if not cfg.path("raw_csv").exists():
    raise FileNotFoundError("Raw data not found. Run: python -m air_quality.cli download  (requires OPENWEATHER_API_KEY in .env)")'''

# ------------------------------------------------------------------ 01
write("01_dataset_audit.ipynb", [
    md("""# 01 — Dataset audit
**Project:** Next-day PM2.5 forecasting for Astana, Kazakhstan (OpenWeatherMap Air Pollution API)

This notebook loads the *actual* downloaded raw dataset, checks its provenance metadata, schema, timestamps,
completeness, duplicates and missing values. Nothing here is simulated: if the raw file does not exist the
notebook stops and tells you which command to run.

Data products: `data/raw/openweather_astana_2021_2025.csv` (raw hourly), `data/raw/dataset_metadata.json`,
`data/raw/chunks/manifest.json` (download log)."""),
    code(SETUP),
    code(GUARD),
    md("## 1. Provenance metadata\nThe metadata file is written by the downloader and records the endpoint, coordinates, requested interval, number of API requests and record counts."),
    code('''meta = json.loads(cfg.path("raw_metadata").read_text(encoding="utf-8"))
print(json.dumps({k: meta[k] for k in ["provider", "endpoint", "location", "api_returned_coordinates",
                                       "requested_interval_utc", "n_records", "first_timestamp_utc",
                                       "last_timestamp_utc", "units", "note"]}, indent=2))
print("\\nDownload summary:", json.dumps(meta.get("download"), indent=2, default=str))'''),
    md("## 2. Download manifest\nEach chunk is recorded as `ok`, `split` (re-requested in smaller pieces) or `failed`. Failed chunks are never silently ignored."),
    code('''man = json.loads(cfg.path("manifest").read_text(encoding="utf-8"))["chunks"]
mf = pd.DataFrame(man.values())
display(mf["status"].value_counts().rename("chunks").to_frame())
ok = mf[mf.status == "ok"]
print(f"Chunks OK: {len(ok)} | records in OK chunks: {int(ok.n_records.sum())} | "
      f"mean chunk coverage: {100*ok.coverage.mean():.2f}% | chunks below 90% coverage: {(ok.coverage < 0.9).sum()}")
display(ok.sort_values("coverage").head(10)[["start_utc", "end_utc", "n_records", "expected", "coverage"]])'''),
    md("## 3. Raw table: structure and data types"),
    code('''raw = pd.read_csv(cfg.path("raw_csv"))
print(raw.shape)
display(raw.head())
display(raw.dtypes.rename("dtype").to_frame())'''),
    md("## 4. Audit statistics (timestamps, coverage, missingness, PM2.5 range)"),
    code('''from air_quality.data_validation import audit_hourly, format_audit
a = audit_hourly(raw, cfg.tz, min_valid_hours=cfg["forecast"]["target_min_valid_hours"])
print(format_audit(a, "Raw dataset audit"))'''),
    code('''miss = pd.DataFrame({"missing": a["missing_by_field"], "missing_pct": a["missing_pct_by_field"]})
display(miss)
print(f"Interpretation: the raw file spans {a['first_timestamp_local']} to {a['last_timestamp_local']} (local). "
      f"It contains {a['n_records']} records for {a['expected_hourly_intervals']} expected hourly slots, i.e. "
      f"{a['coverage_pct']:.2f}% timestamp coverage ({a['missing_hours']} missing hours). "
      f"{a['n_valid_days_min_20h']} of {a['n_calendar_days_local']} local days have at least 20 hourly records.")'''),
    md("## 5. Coverage by year and month"),
    code('''ts = pd.to_datetime(raw["dt_unix"], unit="s", utc=True).dt.tz_convert(cfg.tz)
cov = ts.groupby([ts.dt.year, ts.dt.month]).size().unstack(0)
hours_in_month = pd.DataFrame({y: [pd.Period(f"{y}-{m:02d}").days_in_month * 24 for m in range(1, 13)] for y in cov.columns}, index=range(1, 13))
display((100 * cov / hours_in_month).round(1).rename_axis(index="month", columns="year"))'''),
    md("## 6. Units and value ranges\nAccording to the official API documentation all `components` are concentrations in µg/m³ and `main.aqi` is the provider's 1–5 index. The table below checks the observed ranges (negative values would be physically impossible)."),
    code('''display(raw[["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]].describe().T.round(3))
print("Negative values per field:", a["n_negative_values"])'''),
    md("""## 7. Notes on data credibility
* The values are **API-provided concentrations for a coordinate** (51.1694 N, 71.4491 E), produced by the
  provider's modelling/aggregation; they are not readings from a specific ground monitor and are not the
  average over the whole city.
* No ground-station comparison was performed in this project, so no claim of physical validation is made.
* All timestamps are Unix seconds (UTC) at the **start** of each hour and converted to Asia/Almaty for daily
  aggregation. Kazakhstan moved to a single UTC+5 zone on 2024-03-01; the IANA database handles this change."""),
])

# ------------------------------------------------------------------ 02
write("02_exploratory_data_analysis.ipynb", [
    md("""# 02 — Exploratory data analysis
Statistical exploration of the cleaned hourly grid (`data/interim/hourly_air_pollution_clean.csv`) and the daily
targets. Figures are produced by `air_quality.eda.run_eda` and saved to `artifacts/figures/`; this notebook
re-computes the key statistics visibly and prints data-driven interpretations."""),
    code(SETUP),
    code('''if not cfg.path("features_csv").exists():
    raise FileNotFoundError("Run: python -m air_quality.cli preprocess")
from air_quality.preprocessing import load_clean
from air_quality.eda import run_eda, SEASONS
g = load_clean(cfg.path("clean_hourly_csv"), cfg.tz)
obs = g[g.is_observed].copy()
summary = run_eda(cfg)   # writes all EDA figures + eda_summary.json
print(f"Grid rows: {len(g)} | observed hours: {len(obs)} | missing hours: {(~g.is_observed).sum()}")'''),
    md("## 1. Dataset overview and descriptive statistics (µg/m³)"),
    code('''desc = pd.read_csv(MET / "eda_descriptive_statistics.csv", index_col=0)
display(desc.round(2))'''),
    md("## 2. Missing values and completeness over time"),
    code('''display(Image(FIG / "02_missing_values_completeness.png"))
mc = pd.read_csv(MET / "eda_monthly_completeness.csv")
print(f"Months with <90% observed hours: {(mc.pct_hours_observed < 90).sum()} of {len(mc)}; "
      f"longest gap: {summary['gaps']['max_gap_hours']} h; number of gaps: {summary['gaps']['n_gaps']}")'''),
    md("## 3. Distributions of PM2.5, PM10 and other pollutants"),
    code('''display(Image(FIG / "03b_pm25_distribution.png")); display(Image(FIG / "03_pollutant_distributions.png"))
print(f"Hourly PM2.5 skewness = {obs.pm2_5.skew():.2f}; median = {obs.pm2_5.median():.2f}, mean = {obs.pm2_5.mean():.2f} µg/m³ "
      "(mean > median indicates a right-skewed distribution).")'''),
    md("## 4. Hourly and daily PM2.5 time series"),
    code('''display(Image(FIG / "01_pm25_timeseries.png"))'''),
    md("## 5. Monthly, seasonal, day-of-week and hour-of-day patterns\nHeating season is defined by convention as **15 October – 15 April** (documented in `config.yaml`; it approximates the municipal heating period and is an assumption, not a measured quantity)."),
    code('''display(Image(FIG / "04_seasonal_patterns.png")); display(Image(FIG / "04b_monthly_trend_by_year.png"))
sm = pd.Series(summary["seasonal_mean"]).reindex(["Winter", "Spring", "Summer", "Autumn"])
display(sm.rename("mean PM2.5 (µg/m³)").to_frame())
hm = pd.DataFrame(summary["heating_mean"])
display(hm)
ratio = hm.loc["Heating", "mean"] / hm.loc["Non-heating", "mean"]
print(f"Interpretation: mean hourly PM2.5 in the heating season is {ratio:.2f} times the non-heating mean. "
      "This is an association in the API data; the cause is not established by this analysis.")'''),
    md("## 6. Correlations and PM2.5 versus other pollutants"),
    code('''display(Image(FIG / "05_correlation_heatmap.png")); display(Image(FIG / "05b_pm25_vs_pollutants.png"))
c = pd.read_csv(MET / "eda_spearman_correlation.csv", index_col=0)["pm2_5"].drop("pm2_5").sort_values(key=abs, ascending=False)
display(c.rename("Spearman rho with PM2.5").to_frame().round(3))'''),
    md("## 7. Outliers and extreme pollution\nExtreme hours are inspected statistically and **retained** (high-pollution episodes are exactly what a forecaster must handle)."),
    code('''print(json.dumps(summary["extremes"], indent=2))'''),
    md("## 8. Autocorrelation, trend and stationarity"),
    code('''display(Image(FIG / "06_autocorrelation.png"))
print("Hourly ACF:", summary["acf_hourly"]); print("Daily ACF:", summary["acf_daily"]); print("ADF (daily):", summary["adf_daily"])
print(f"Interpretation: the lag-1 daily autocorrelation is {summary['acf_daily']['1']:.2f}, which motivates lagged and rolling "
      "features and makes persistence a strong baseline.")'''),
    md("## 9. Train / validation / test split and distribution shift"),
    code('''display(Image(FIG / "07_split_timeline.png")); display(Image(FIG / "07b_split_distribution_shift.png"))
display(pd.read_csv(MET / "eda_split_distribution_shift.csv", index_col=0).round(2))
print("Kolmogorov–Smirnov train vs other splits:", summary["ks_train_vs"])'''),
])

# ------------------------------------------------------------------ 03
write("03_preprocessing_and_features.ipynb", [
    md("""# 03 — Cleaning, target construction, feature engineering, normalization and feature selection
This notebook shows every preprocessing step and demonstrates the leakage controls.

**Forecast protocol.** Issue time = 18:00 Asia/Almaty on day *d*; target = mean PM2.5 over local day *d+1*
(≥ 20 valid hours). Records are labelled by the start of their hour, so at 18:00 the newest usable record is
the one labelled 17:00."""),
    code(SETUP),
    code(GUARD),
    md("## 1. Cleaning (global, split-independent operations)"),
    code('''from air_quality.preprocessing import clean_hourly
raw = pd.read_csv(cfg.path("raw_csv"))
clean, rep = clean_hourly(raw, cfg.tz)
for s in rep["steps"]:
    print(s)
print("\\nGaps:", rep["gaps"]); print("Extremes:", rep["extremes"])'''),
    code('''q = json.loads(cfg.path("quality_report_json").read_text())
cmp = pd.DataFrame({"before": q["before"], "after": q["after"]}).loc[
    ["n_records", "n_unique_timestamps", "n_duplicated_timestamps", "coverage_pct", "missing_hours", "pm2_5_min", "pm2_5_max", "n_valid_days_min_20h"]]
display(cmp)'''),
    md("## 2. Daily target construction\n$y(d) = \\frac{1}{|H_{d+1}|}\\sum_{h \\in H_{d+1}} \\mathrm{PM2.5}_h$, valid only if $|H_{d+1}| \\geq 20$."),
    code('''from air_quality.feature_engineering import daily_targets, build_dataset, compute_features, issue_time_for_date, FEATURE_SETS
t = daily_targets(clean, cfg["forecast"]["target_min_valid_hours"])
print(f"Local days: {len(t)} | valid targets: {int(t.target_is_valid.sum())} | excluded (<20 h): {int((~t.target_is_valid).sum())}")
display(t.target_valid_hours.value_counts().sort_index().rename("days").to_frame().T)'''),
    md("## 3. Features computed at the issue time — worked example"),
    code('''ds = pd.read_csv(cfg.path("features_csv"))
ex_date = ds[ds.eligible & (ds.split == "test")].issue_date.iloc[10]
issue = issue_time_for_date(ex_date, cfg.tz, cfg["forecast"]["issue_hour_local"])
fc = {**cfg["forecast"]}
row = compute_features(clean, [issue], fc, cfg["evaluation"]["heating_season"]).iloc[0]
cut = issue.tz_convert("UTC")
w = clean[(clean.timestamp_utc < cut) & (clean.timestamp_utc >= cut - pd.Timedelta(hours=24))]
print(f"Issue time {issue} -> target date {row.target_date}")
print(f"Window used for pm25_mean_24h: {w.timestamp_local.min()} .. {w.timestamp_local.max()} (newest label < issue)")
print(f"pm25_mean_24h = {row.pm25_mean_24h:.3f} | manual mean = {w.pm2_5.mean():.3f}")
display(row[FEATURE_SETS["full"]].to_frame("value").T)'''),
    md("## 4. Leakage checks (executed)\n1. Overwrite every observation at/after the issue time with an absurd value — features must not change.\n2. Overwrite the target day — features must not change.\n3. Every row's newest record label + 1 h ≤ issue time."),
    code('''g2 = clean.copy(); g2.loc[g2.timestamp_utc >= cut, ["pm2_5", "pm10", "no2", "co", "o3", "so2", "nh3", "no", "aqi"]] = 1e6
row2 = compute_features(g2, [issue], fc, cfg["evaluation"]["heating_season"]).iloc[0]
same = all((np.isnan(row[f]) and np.isnan(row2[f])) or np.isclose(row[f], row2[f]) for f in FEATURE_SETS["full_aqi"])
newest = pd.to_datetime(ds.meta_newest_record_utc, utc=True); iss = pd.to_datetime(ds.issue_time_utc, utc=True)
ok_rows = ((newest + pd.Timedelta(hours=1)) <= iss)[newest.notna()].all()
print("Check 1/2 (future values ignored):", "PASSED" if same else "FAILED")
print("Check 3 (all rows use only past records):", "PASSED" if ok_rows else "FAILED")'''),
    md("## 5. Feature sets used in the experiments"),
    code('''for k, v in FEATURE_SETS.items():
    print(f"{k:<9} {len(v):>3} features")
print("full_selected = full + in-pipeline selection (VarianceThreshold -> CorrelationFilter -> TopK f_regression)")
el = ds[ds.eligible]
display(el.groupby("split").agg(n=("target_pm25", "size"), first=("target_date", "min"), last=("target_date", "max"),
                                mean_target=("target_pm25", "mean")).round(2))'''),
    md("## 6. Training-only correlation analysis and feature selection"),
    code('''from air_quality.feature_selection import training_correlation_table, make_selector
from sklearn.impute import SimpleImputer
tr = el[el.split == "train"]
ct = training_correlation_table(tr, FEATURE_SETS["full"], "target_pm25")
display(ct.head(15).round(3))
Xtr = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(tr[FEATURE_SETS["full"]])
sel = make_selector(k=20).fit(Xtr, tr.target_pm25)
names = np.array(FEATURE_SETS["full"])
for _, step in sel.steps:
    print(f"{type(step).__name__:<18} keeps {step.get_support().sum():>3} of {len(names)}")
    names = names[step.get_support()]
print("Selected (k=20, illustration; k is tuned during training):", list(names))'''),
    md("## 7. Normalization\n`StandardScaler` is applied only inside the Ridge and SVR pipelines, after median imputation; both are fitted on training folds only. Tree models are scale-invariant and are not scaled."),
    code('''from air_quality.models import build_pipeline
p = build_pipeline("svr").fit(tr[FEATURE_SETS["basic"]], tr.target_pm25)
print(p)
print("Scaler means learned from TRAIN only (first 5):", np.round(p.named_steps["scale"].mean_[:5], 3))'''),
])

# ------------------------------------------------------------------ 04
write("04_model_training.ipynb", [
    md("""# 04 — Model training and hyperparameter optimization
Four regression models + persistence baseline. All preprocessing lives inside scikit-learn pipelines.

| # | Model | Pipeline |
|---|---|---|
| 1 | Ridge Regression | median imputer → [selector] → StandardScaler → Ridge |
| 2 | Random Forest | median imputer → [selector] → RandomForestRegressor |
| 3 | XGBoost | median imputer → [selector] → XGBRegressor |
| 4 | SVR (RBF) | median imputer → [selector] → StandardScaler → SVR |
| – | Persistence | ŷ = trailing 24-h mean PM2.5 at 18:00 |

Hyperparameters: `RandomizedSearchCV` on the 2021–2023 training targets with `TimeSeriesSplit` (gap = 1 day),
scoring = MAE. Selection by 2024 validation MAE. Final refit on train + validation; the 2025 test is used once."""),
    code(SETUP),
    code('''from air_quality.models import build_pipeline, param_space, MODEL_NAMES
from air_quality.training import run_training, load_model_frame, make_cv
df = load_model_frame(cfg)
print(df.groupby("split").size())
for m in MODEL_NAMES:
    print(f"\\n{m}:", build_pipeline(m, select=False))
    print("  search space:", {k: (v if isinstance(v, list) else f"{type(v).__name__}") for k, v in param_space(m).items()})
print("\\nCV:", make_cv(cfg, (df.split == "train").sum()))'''),
    md("## Run training\nSet `RUN_TRAINING = False` to reuse frozen results in `artifacts/` (recommended once the benchmark is frozen)."),
    code('''RUN_TRAINING = not (cfg.path("metrics_dir") / "all_results.csv").exists()
if RUN_TRAINING:
    out = run_training(cfg)          # full budget from config.yaml (n_iter per model/config)
res = pd.read_csv(MET / "all_results.csv")
print("Rows in all_results.csv:", len(res))'''),
    md("## Cross-validation and validation results (tuned models)"),
    code('''v = res[(res.stage == "validation")].pivot_table(index=["model", "feature_config"], columns="variant", values=["cv_mae", "mae", "rmse", "r2"])
display(v.round(3))'''),
    md("## Selected hyperparameters"),
    code('''bp = res[(res.variant == "tuned") & (res.stage == "validation")][["model", "feature_config", "best_params", "train_seconds", "n_features_used"]]
display(bp)'''),
    md("## Model selection (validation only)"),
    code('''sel = json.loads((MET / "model_selection.json").read_text())
print(json.dumps(sel, indent=2))'''),
])

# ------------------------------------------------------------------ 05
write("05_model_evaluation.ipynb", [
    md("""# 05 — Independent test evaluation, experiments E1–E7, interpretability
All numbers come from the frozen benchmark predictions in `artifacts/predictions/` (2025 targets never used for
any fitting or selection)."""),
    code(SETUP),
    code('''from air_quality.evaluation import run_evaluation
from air_quality.explainability import run_explainability
ev = run_evaluation(cfg); ex = run_explainability(cfg)
comp = pd.read_csv(MET / "model_comparison.csv")
display(comp[["model", "feature_config", "validation_mae", "test_mae", "test_rmse", "test_r2", "test_smape_pct",
              "test_bias", "test_mae_improvement_pct", "test_n"]].round(3))'''),
    md("## E1 — Model comparison (same eligible test dates)"),
    code('''display(Image(FIG / "11_model_comparison.png"))
best = ev["selection"]["best"]; b = comp[comp.model == best["model"]].iloc[0]; p = comp[comp.model == "persistence"].iloc[0]
print(f"The validation-selected model is {best['model']} ({best['feature_config']}). On the 2025 test it obtained "
      f"MAE = {b.test_mae:.2f} µg/m³ (RMSE {b.test_rmse:.2f}, R² {b.test_r2:.3f}) compared with "
      f"MAE = {p.test_mae:.2f} µg/m³ for persistence ({b.test_mae_improvement_pct:.1f}% change in MAE).")
lowest = comp.loc[comp.test_mae.idxmin()]
print(f"Lowest test MAE overall: {lowest.model} ({lowest.test_mae:.2f}); reported for transparency only — selection used validation.")'''),
    md("## E2 — Effect of feature engineering (basic → lag → lag + rolling)"),
    code('''e2 = pd.read_csv(MET / "E2_feature_engineering.csv"); display(e2.round(3))
display(Image(FIG / "13_ablation_feature_configs.png"))'''),
    md("## E3 — Feature selection"),
    code('''display(pd.read_csv(MET / "E3_feature_selection.csv").round(3))'''),
    md("## E4 — Default vs tuned hyperparameters"),
    code('''display(pd.read_csv(MET / "E4_hyperparameter_optimization.csv").round(3))'''),
    md("## E5 — Elevated-pollution days\nThresholds: (i) 90th percentile of **training** daily targets; (ii) WHO (2021) 24-h AQG level of 15 µg/m³."),
    code('''e5 = pd.read_csv(MET / "E5_elevated_pollution.csv"); display(e5.round(3))
display(Image(FIG / "15_elevated_pollution_predictions.png"))
n = e5[e5.threshold_name.str.startswith("train")].n_elevated_days.iloc[0]
print(f"Number of elevated test days (train q90): {n}. {'Small sample — interpret with caution.' if n < 30 else ''}")'''),
    md("## E6 — Interpretability (associational, not causal)"),
    code('''display(Image(FIG / "12_feature_importance_permutation.png"))
for f in ["12b_feature_importance_trees.png", "12c_ridge_coefficients.png", "12d_shap_summary.png"]:
    if (FIG / f).exists(): display(Image(FIG / f))
print("Top features (best model, permutation on 2025 test):", ex["top_permutation_features_best_model"])'''),
    md("## E7 — Robustness across months, seasons and heating periods"),
    code('''display(Image(FIG / "14_monthly_error.png"))
display(pd.read_csv(MET / "E7_seasonal_errors.csv").pivot(index="season", columns="model", values="mae").round(2))
display(pd.read_csv(MET / "E7_heating_errors.csv").pivot(index="heating", columns="model", values="mae").round(2))'''),
    md("## Error analysis"),
    code('''for f in ["08_actual_vs_predicted.png", "08b_test_timeseries.png", "09_residual_distribution.png", "10_residuals_vs_predicted.png"]:
    display(Image(FIG / f))
pt = pd.read_csv(cfg.path("predictions_dir") / "predictions_test.csv")
bt = pt[(pt.model == best["model"]) & (pt.feature_config == best["feature_config"]) & (pt.variant == "tuned")]
display(bt.nlargest(10, "abs_error")[["issue_time_local", "target_date", "actual", "predicted", "abs_error"]].round(2))'''),
    md("## Hypothesis evaluation (automatically derived from the results)"),
    code('''from air_quality.report import hypothesis_verdicts
for k, v in hypothesis_verdicts(cfg).items():
    print(f"{k}: {v}")'''),
])
print("notebooks written:", sorted(p.name for p in OUT.glob("*.ipynb")))
