# Discussion of results

*Benchmark run of 17 November 2026 with five models and seven feature configurations
(`artifacts/metrics/`). All numbers below are read from `model_comparison.csv`, `all_results.csv`,
`E5_elevated_pollution.csv`, `E6_summary.json` and `E7_heating_errors.csv`.*

**Data.** 43,273 hourly OpenWeatherMap records (26 Dec 2020 – 31 Dec 2025 local; 98.4% timestamp coverage,
no duplicates) and hourly Open-Meteo weather for the same period. After eligibility filtering: 1,061 training
(2021–2023), 351 validation (2024) and 346 test (2025) daily samples. Mean hourly PM2.5 was 7.85 µg/m³ in the
heating season and 4.06 µg/m³ outside it; the 2025 test year was cleaner than the training years (median daily
target 2.76 vs 4.47 µg/m³; KS p < 0.001).

**Model selection (validation 2024).** Adding weather produced the largest improvement of the whole study on
the validation year: the best validation MAE fell from 3.070 µg/m³ without weather (Ensemble, *full_selected*)
to **2.227 µg/m³ with weather (Ensemble, *full_weather*)**, a 27% reduction that is consistent with the ~36% RMSE
reduction reported by Domalatov et al. (2025) for Almaty. All five models improved on validation with weather
(e.g. XGBoost 3.122 → 2.249, SVR 3.075 → 2.261, Random Forest 3.094 → 2.352). Following the protocol, the
**Ensemble (equal-weight RF + XGBoost + SVR) with *full_weather*** was selected and frozen.

**Independent test (2025).**

| Model (selected configuration) | Test MAE | Test RMSE | Test R² | Better than persistence (MAE) |
|---|---|---|---|---|
| **Ensemble, full_weather (selected)** | **3.015** | **4.655** | **0.414** | **14.2%** |
| SVR, full_weather | 2.959 | 4.665 | 0.411 | 15.8% |
| XGBoost, full_weather | 3.148 | 4.768 | 0.385 | 10.5% |
| Random Forest, full_weather | 3.120 | 4.793 | 0.378 | 11.2% |
| Ridge, full_weather | 3.715 | 5.271 | 0.248 | −5.7% |
| Persistence | 3.516 | 6.059 | 0.006 | — |

The large validation gain did **not** transfer fully to 2025. Compared with the best model without weather
in the same run (SVR, *full_selected*: test MAE 2.925, RMSE 5.017, R² 0.319), the selected weather ensemble has
a slightly higher MAE (+0.09 µg/m³) but a clearly lower RMSE (−0.36 µg/m³) and higher R² (0.414 vs 0.319): it
makes fewer large errors and slightly more small ones. The reason is visible in E7: in the non-heating season
the weather models over-predict (bias +1.5 µg/m³ for the ensemble, MAE 1.85 vs 1.48 for persistence), while in
the heating season they are much better than persistence (MAE 4.27 vs 5.70). A plausible explanation is the
cleaner 2025 summer, so the same weather produced less PM2.5 than in 2021–2024. A second is that target-day
weather in training is reanalysis (perfect prognosis). The strictly operational configuration with only past
weather (*full_weather_past*) gave the most stable results: Ensemble test MAE 2.942, Random Forest R² 0.433
(the highest test R² of the study). Test results were not used to change any choice.

**Hypotheses.**
* **H1** (a tree ensemble is best) — *partly supported*: the selected model is the ensemble of Random Forest,
  XGBoost and SVR; among single models with weather, XGBoost had the best validation MAE (2.249).
* **H2** (engineered features help) — *supported*: lag/rolling features lowered validation MAE for every model
  relative to the basic set, and weather features lowered it further for every model.
* **H3** (ML beats persistence) — *supported*: the selected model reduces test MAE by 14.2% (3.516 → 3.015) and
  RMSE by 23% (6.06 → 4.66); four of five models beat persistence (Ridge with weather does not).
* **H4** (recent PM2.5 dominates) — *supported*: top permutation features of the selected model are the last
  PM2.5 value (0.45 µg/m³ MAE increase) and last PM10 (0.26), followed by seasonality and **target-day wind
  speed, number of calm hours, temperature change and pressure change** (0.08–0.11) — the meteorological
  conditions known to trap pollution.
* **H5** (larger errors on episodes) — *supported*, but weather helped: on the 33 test days above the training
  90th percentile (13.2 µg/m³) the selected model's MAE was 9.94 µg/m³ (12.68 for the pollution-only SVR selected in the first run)
  versus 2.29 on other days, with a negative bias of −9.6 µg/m³. For days above the WHO 15 µg/m³ guideline
  (n = 23) the best model was SVR with weather (MAE 10.92), now better than persistence (11.34).

**Interpretation.** Meteorology is the most valuable addition: it improves validation accuracy for every
model, reduces large errors and improves forecasts of pollution episodes, in line with Domalatov et al. (2025).
Year-to-year changes in the relationship between weather and PM2.5 (the clean 2025 summer) limit the
improvement in mean absolute error on the test year, which shows why an independent test year is necessary.
Day-ahead accuracy (R² ≈ 0.41) remains far below concurrent-estimation studies (R² ≈ 0.98; Makhdoomi et al.,
2025), which answer a different question.

**Live demonstration.** On 17 Nov 2026 (before 18:00, `--issue-mode latest`) the system used the selected
ensemble with the Open-Meteo weather forecast and issued 3.32 µg/m³ for 17 Nov 2026 (forecast issued 16 Nov
18:00). The OpenWeatherMap provider forecast for the same day was 3.21 µg/m³ (reference only).
