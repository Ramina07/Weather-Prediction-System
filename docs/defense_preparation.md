# Defense preparation

## 1. Team contribution plan (proposed — to be confirmed by the team)

This is a proposed division of responsibilities, not a record of completed work. Each member must be able to
explain the whole system. Replace with the actual contributions before submission; any non-contributing member
must be stated in the report.

| Area | Member 1 — Data engineering & analysis | Member 2 — ML & experimentation | Member 3 — Research integration & reproducibility |
|---|---|---|---|
| Code ownership | `api_client.py`, `download.py`, `data_validation.py`, `preprocessing.py`, `eda.py` | `feature_engineering.py`, `feature_selection.py`, `models.py`, `training.py`, `evaluation.py`, `explainability.py` | `prediction.py`, `cli.py`, `report.py`, `tests/` |
| Notebooks | 01, 02 | 03, 04, 05 | execution of all notebooks, consistency with report |
| Documents | `data_provenance.md`, `data_dictionary.md` | `mathematical_formulations.md`, experiment sections | `literature_review.md`, `literature_comparison.*`, `research_methodology.md`, report, slides |
| Defense topic | Data source, quality, cleaning, EDA | Models, tuning, results, interpretability | Literature, leakage controls, demo, limitations |

## 2. Questions and answers

**Dataset**

*Q: Where do the data come from and how reliable are they?*
A: From the OpenWeatherMap Air Pollution API history endpoint for one coordinate in Astana. They are
API-provided concentrations, not ground-station readings; the provider does not document the production method
on the API page. We measured completeness, duplicates and missingness ourselves (Table 1 of the report) and we
did not validate them against stations, so we only claim to forecast this API series.

*Q: Why 18:00 and why the daily mean?*
A: A fixed issue time makes the information set unambiguous and reproducible. 18:00 is late enough to include
most of the current day and early enough to be useful for planning tomorrow. The daily mean matches the WHO
24-hour guideline (15 µg/m³).

*Q: What happens with missing hours?*
A: They stay missing; we never interpolate the hourly series. Window features need ≥ 75% valid hours, a target
day needs ≥ 20 of 24 hours, otherwise it is excluded. Missing feature values are filled with the median of the
training fold inside the pipeline.

**Leakage**

*Q: How do you guarantee no future information is used?*
A: (1) Records are labelled by the start of their hour, so at 18:00 the newest usable record is 17:00. (2) Every
feature is computed by one function that only reads indices up to that record. (3) Tests overwrite all values at
and after the issue time, and the whole target day, with absurd numbers and check that no feature changes.
(4) Splits are by target date. (5) Imputer, scaler and selector are fitted inside the pipeline on training folds.

*Q: Why is persistence the trailing 24-h mean and not yesterday's daily mean?*
A: Today's calendar-day mean would need observations after 18:00. The trailing 24 hours ending at 17:00 are
fully available at the issue time.

*Q: Why a gap in TimeSeriesSplit?*
A: The target day of one sample is the issue day of the next sample, so adjacent samples share information. A
1-day embargo between CV-training and CV-validation folds removes this overlap.

**Models**

*Q: Why these four models?*
A: They represent four different hypothesis classes that appear in recent PM2.5 studies: regularised linear
(Ridge), bagged trees (Random Forest), boosted trees (XGBoost) and kernel regression (SVR). The literature shows
no single winner across settings, so we compare them on identical dates.

*Q: Explain Ridge / RF / XGBoost / SVR in one sentence each.*
A: Ridge minimises squared error plus α‖β‖² to keep coefficients stable; Random Forest averages many trees grown
on bootstrap samples with random feature subsets; XGBoost adds trees sequentially, each fitted to the gradient of
the loss with L2-regularised leaf weights; SVR fits a function that ignores errors smaller than ε and penalises
larger ones linearly, with an RBF kernel for non-linearity.

*Q: Why scale only Ridge and SVR?*
A: Both depend on distances or coefficient magnitudes; tree splits are invariant to monotone rescaling.

*Q: How were hyperparameters chosen?*
A: RandomizedSearchCV with 25 candidates on 2021–2023 using time-series CV and MAE; the configuration with the
lowest 2024 validation MAE was selected; the test year 2025 was used once.

**Performance**

*Q: Did ML beat persistence?* A: Answer with H3 from Section 4.8 of the generated report (actual numbers).

*Q: Is R² = 0.x "x% accuracy"?* A: No. R² is the share of target variance explained relative to predicting the
mean; it can be negative and is not a percentage of correct predictions.

*Q: Why is your error higher than in some published papers?*
A: Many high scores come from concurrent estimation (inputs measured at the same time as the target) or very
short horizons and random splits. Our task is a genuine day-ahead forecast with a chronological test year.

*Q: Where does the model fail?* A: Use E5/E7: elevated days and transitions into/out of episodes; errors are
largest in the heating season.

**Practical use**

*Q: Show the program working.* A: `python -m air_quality.cli predict` (after 18:00) or
`--issue-mode latest` (before 18:00); show the JSON in `artifacts/predictions/`.

*Q: How is the provider's own forecast used?* A: Only optionally, labelled "OpenWeatherMap Provider Forecast";
it is never presented as our model and was not used for historical evaluation.

*Q: What would you improve?* A: Meteorological forecasts as inputs, ground-station validation, rolling-origin
evaluation, probabilistic forecasts and a GUI on top of `prediction.make_forecast`.
