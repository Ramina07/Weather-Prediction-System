# Research methodology

## 1. Problem definition

Supervised regression / time-series forecasting. For each local date $d$ a forecast is issued at
**18:00 Asia/Almaty** and predicts $y_d$, the **mean PM2.5 (µg/m³) over local calendar day $d+1$**, from
information available at the issue time only. Classification metrics are not used.

Example: forecast issued 2025-06-10 18:00 → features use records labelled up to 2025-06-10 17:00 → target is
the mean of the hourly PM2.5 records of 2025-06-11 (00:00–23:00 labels).

## 2. Aim, objectives, research questions and hypotheses

**Aim.** Develop and evaluate a reproducible ML system that predicts next-day PM2.5 in Astana from historical
OpenWeatherMap air-pollution data.

| Objective | Implemented in |
|---|---|
| RO1 Collect and validate reliable historical data | `download.py`, `data_validation.py`, notebook 01 |
| RO2 Analyse temporal patterns, seasonality, correlations, missingness | `eda.py`, notebook 02 |
| RO3 Leakage-free feature engineering and preprocessing | `preprocessing.py`, `feature_engineering.py`, `feature_selection.py`, `tests/test_leakage.py`, notebook 03 |
| RO4 Implement and optimise four regression algorithms | `models.py`, `training.py`, notebook 04 |
| RO5 Compare against each other and persistence on chronological test | `training.py`, `evaluation.py` (E1), notebook 05 |
| RO6 Feature importance, error analysis, limitations | `explainability.py` (E6), `evaluation.py` (E5, E7) |
| RO7 Executable next-day prediction | `prediction.py`, `cli.py predict` |
| RO8 Documentation and link to literature | `docs/` |

| RQ | Hypothesis (to be tested, not assumed) | Test |
|---|---|---|
| RQ1 Which model gives the lowest next-day error? | H1: a tree ensemble (RF or XGBoost) achieves the lowest validation MAE. | E1 (validation selection, test confirmation) |
| RQ2 Do engineered historical features help vs basic features? | H2: *lag* and *full* feature sets reduce MAE relative to *basic*. | E2 |
| RQ3 Does ML beat persistence? | H3: the selected model has lower test MAE than persistence (skill > 0). | E1 |
| RQ4 Which pollutants/temporal features contribute most? | H4: recent PM2.5 history (last value, rolling means) dominates importance; seasonal calendar features rank next. | E6 |
| RQ5 How robust are models on elevated days? | H5: errors on elevated days are substantially larger than on other days, and models under-predict them (negative bias). | E5, E7 |

## 3. Data

* **Source:** OpenWeatherMap Air Pollution API, history endpoint `/data/2.5/air_pollution/history`
  (documented availability from 27 Nov 2020). Values are API-provided concentrations for the coordinate
  51.1694 N, 71.4491 E; they are not direct readings from a specific ground station and not a city-wide mean.
* **Period:** local dates 2021-01-01 to 2025-12-31 (+6-day lead-in for lag features). 2026 data are
  downloaded separately (`update`) for operational retraining only.
* **Acquisition:** 7-day UTC chunks, automatic halving when a chunk returns < 50% of expected hours, retry
  with exponential backoff (429/5xx/timeouts), cached raw JSON, manifest, resumable.
* **Units:** µg/m³ for all components; AQI is the provider's 1–5 index (OpenWeather, n.d.).

## 4. Cleaning (global, split-independent)

Schema/type validation → drop invalid and future timestamps → chronological sort → duplicate removal (exact,
then conflicting per timestamp; first kept, counts logged) → negative concentrations and AQI ∉ {1..5} set to
NaN → reindex to a complete hourly UTC grid (missing hours remain NaN, **no interpolation**) → local-time
annotation → robust-z extreme flag (flag only, extremes retained). Report before/after:
`artifacts/metrics/data_quality_report.{json,md}`.

These steps do not use any statistic that differs between splits, so they cannot leak information. Every
**fitted** transformation (median imputation, scaling, feature selection) lives inside the model pipeline and is
fitted on training folds only.

## 5. Target and features

* Target: daily mean of valid hours on $d+1$, valid if ≥ 20 hours (configurable); `target_valid_hours` stored;
  invalid targets excluded, never imputed.
* Timestamp convention: hour-beginning labels; records labelled ≥ 18:00 on day $d$ are never read for issue $d$.
* Feature groups (43 + 2 AQI):
  * **Calendar of the target day** (9): day of week, month, season, weekend, sin/cos month, sin/cos day-of-year, heating-season flag (15 Oct–15 Apr).
  * **Basic history** (1): last available PM2.5 (≤ 3 h old).
  * **Lags** (14): PM2.5 at 24/48/72 h before the last hour; last value of PM10, NO2, CO, O3, SO2, NH3, NO; 24-h lags of PM10, NO2, CO, O3.
  * **Rolling** (19): PM2.5 trailing means (6/12/24/48/72 h), 24-h SD/min/max, 24-h OLS trend, difference of consecutive 24-h means, previous local-day mean, PM2.5/PM10 ratio, 24-h means of the other seven pollutants.
  * **AQI** (2, ablation only): last AQI, 24-h mean AQI.
* Feature configurations: `basic` (10), `lag` (24), `full` (43), `full_selected` (43 → in-pipeline
  selection), `full_aqi` (45, ablation, excluded from model selection).
* Eligibility (identical for all models and the baseline): valid target, `pm25_last` and `pm25_mean_24h` available.

## 6. Splits and validation

* Split by **target date**: train 2021-01-01..2023-12-31, validation 2024, test 2025. Earlier observations may
  feed lag features of later samples (they were observable before those issue times).
* Hyperparameter search: `RandomizedSearchCV` (25 candidates per model × configuration, seed 42) with
  `TimeSeriesSplit(n_splits=5, gap=1)` on training samples, scoring = MAE. The 1-day gap is a conservative
  embargo: the target day of a CV-training sample is the issue day of the next sample.
* Selection: lowest 2024 validation MAE among tuned models in the main configurations.
* Final benchmark: each model's selected configuration is refitted on train + validation with the same
  hyperparameters and evaluated **once** on 2025. Results are frozen; no tuning is done after looking at 2025.
* Operational model: optional refit of the selected configuration on all labelled data (incl. 2026), stored
  separately (`artifacts/models/operational/`) and never used to report benchmark results.

## 7. Models

Model 5 (added after the literature update): equal-weight ensemble of the tuned Random Forest, XGBoost and SVR (VotingRegressor; no weights fitted on validation/test). Weather configurations `full_weather` (past + target-day weather, perfect prognosis in the benchmark) and `full_weather_past` (strict) — see `weather.py`.


Ridge (L2), Random Forest, XGBoost (hist trees), SVR (RBF) — see `docs/mathematical_formulations.md`. Search
spaces are in `models.param_space` (bounded for a student laptop). Baseline: persistence = trailing 24-h mean.

## 8. Metrics

Primary: MAE. Secondary: RMSE, R², sMAPE, MAPE (targets ≥ 1 µg/m³), bias, skill vs persistence, training and
prediction time, error on elevated days.

## 9. Experiments

| ID | Hypothesis | Subset | Features | Models | Evaluation | Output |
|---|---|---|---|---|---|---|
| E1 | H1, H3 | eligible train/val/test | validation-selected per model | 4 + persistence | val selection, single test | `model_comparison.csv`, Fig. 11 |
| E2 | H2 | same | basic / lag / full | 4 | val + test | `E2_feature_engineering.csv`, Fig. 13 |
| E3 | selection reduces complexity without loss | same | full vs full_selected | 4 | val + test, #features | `E3_feature_selection.csv` |
| E4 | tuning helps | same | all configs | 4 | default vs tuned | `E4_hyperparameter_optimization.csv` |
| E5 | H5 | 2025 test | selected | 4 + persistence | MAE/RMSE/bias above train-q90 and WHO 15 µg/m³ | `E5_elevated_pollution.csv`, Fig. 15 |
| E6 | H4 | 2025 test | selected | 4 | permutation importance, tree importance, Ridge coefficients, optional SHAP | `E6_*.csv`, Fig. 12 |
| E7 | seasonal robustness | 2025 test | selected | 4 + persistence | monthly / seasonal / heating MAE, RMSE | `E7_*.csv`, Fig. 14 |
| Ablation | historical AQI adds little | same | full_aqi | 4 | val + test (not used for selection) | `all_results.csv` |

## 10. Reproducibility

Seed 42 everywhere (`project.random_seed`); `config.yaml` stores every parameter; raw API responses cached;
environment recorded by `python -m air_quality.cli env`; complete run: `python -m air_quality.cli run-all`.
