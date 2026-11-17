# 2. Literature Review

## 2.0. How the review was conducted and how to verify it

**Goal of the review.** Identify which machine learning models perform best in studies that are *similar to
ours* — daily or day-ahead PM2.5 forecasting from historical air-pollution data — and use that evidence to
choose the models we train.

**Selection.** Sixteen candidate studies (2022–2026) were screened; eleven were included after their
metadata and results were checked against the publisher page or an official repository (last verified
17 November 2026). Two candidates could not be retrieved and are not cited (see `literature_comparison.md`).

**How to verify a number yourself.** Every result below has a **"Where to look"** line giving the section,
table, figure or page of the original article. Open the DOI link from the reference list and go to that place.
Some MDPI and PMC tables are images on the web page; in that case the numbers are also stated in the article
text at the location given, or the PDF version shows the table.

**Two kinds of studies.** This distinction decides whether a result is comparable with ours:

* **Forecasting studies** predict PM2.5 for a *future* time using only past data (like our project).
* **Estimation studies** predict PM2.5 at the *same* time from simultaneously measured variables
  (co-pollutants, weather, satellite data). They usually report much higher accuracy, but they answer a
  different question and cannot be used directly for next-day forecasting.

---

## 2.1. Air pollution and PM2.5 forecasting

PM2.5 is the pollutant most closely linked to the health burden of urban air pollution; the World Health
Organization (2021) set its 24-hour guideline level at 15 µg/m³. In Astana, Agibayeva et al. (2022) estimated
that PM2.5 exposure in 2019 corresponded to roughly 2,160–7,531 disability-adjusted life years and found
clear differences between the heating (October–April) and non-heating periods
(*Where to look:* Section 3.7 and Tables 8 and 10 for DALY; Conclusions for the 2,160–7,531 range).

Three findings recur across the reviewed forecasting studies:

1. **PM2.5 is strongly persistent**, so simple "repeat the recent value" forecasts are hard to beat at short
   horizons (Sapakova et al., 2026).
2. **Accuracy drops quickly with the horizon**: XGBoost reached R² = 0.862 one hour ahead but R² = 0.50 at
   24 hours (Fetahi et al., 2025).
3. **Pollution episodes are the hardest to predict** (Haseeb et al., 2026; Agibayeva et al., 2022).

---

## 2.2. Study-by-study results

### A. Forecasting studies (past data → future PM2.5) — most relevant to us

**[L1] Sapakova et al. (2026), Almaty, Kazakhstan — *Technologies* 14(8), 512.**
*Data:* OpenWeather Air Pollution API (the same provider as ours), 5 virtual stations, 18 summer days in 2026,
modelled at 15/30/60-minute steps. *Task:* one-step-ahead nowcasting. *Models:* lag-1 persistence, Linear
Regression, Random Forest, Gradient Boosting; chronological 80/20 split.
*Result:* at 15 minutes persistence had R² = 0.980, while the ML models reached 0.962–0.977; at 60 minutes
all ML models fell to R² ≤ 0.319 against 0.743 for persistence. In Random Forest, the current PM2.5 value
(≈0.72) and its 3-step rolling mean (≈0.24) dominated feature importance.
*Where to look:* Section 3.5 "PM2.5 Forecasting Results", Table 7 (results across 15/30/60 min) and
Figure 9 (Random Forest feature importance). MAE/RMSE values are only in Table 7.
*Meaning for us:* persistence must be included as a baseline; recent PM2.5 and rolling means are the key
features. Not directly comparable: 15-minute horizon, summer only.

**[L3] Fetahi et al. (2025), Prishtina, Kosovo — *Ecological Engineering & Environmental Technology* 26(8), 97–114.**
*Data:* hourly pollution, 2018–2024. *Task:* multi-horizon forecasting. *Models:* XGBoost, LightGBM, Random
Forest, SVM, Linear Regression; lag, rolling and cyclical time features; SHAP.
*Result:* **XGBoost was best**: R² = 0.862, MAE = 3.524, RMSE = 6.513 one hour ahead, and R² = 0.50 at the
24-hour horizon.
*Where to look:* the **Abstract** (all four numbers and the list of five models). Full PDF on the journal page
("Article (PDF)").
*Meaning for us:* the same model families we use; shows the expected accuracy loss at a 24-hour horizon.

**[L4] Psomadakis et al. (2026), Greece — *Electronics* 15(19), 4496.**
*Data:* open data from 5 Greek stations, daily means. *Task:* **next-day** PM2.5/PM10 and next-week maximum.
*Models:* Random Forest, **SVM (RBF)**, neural network, LSTM; persistence and SARIMA baselines.
*Design:* chronological boundaries fixed before gap-filling (Section 4.2); 40 sliding windows for model
selection plus a final 20% held-out set (Section 4.4); persistence baseline defined in Section 4.4.1.
*Result:* **SVM was the most broadly reliable model**, winning 13 of 20 combinations under held-out assessment
and 11 of 20 under model selection. On the held-out data **every ML model beat persistence at every station**
for next-day PM2.5, while SARIMA was worse than persistence at four of five stations. Example: SVM reached
R² = 0.458 at Volos-1.
*Where to look:* Abstract and Section 8 (Conclusions) for 13/20 and 11/20; Section 6.1 and **Table 8** (mean
R² for next-day PM2.5; best model in bold) and **Figure 9** (held-out R² by station and model); Section 6.3 for
the Volos-1 value; Section 7.6.3 for SARIMA vs persistence.
*Meaning for us:* the closest design to ours (next-day daily target, past-only features, persistence
baseline) and the main evidence for including **SVR**.

**[L5] Chivukula et al. (2026), nine U.S. cities — *Atmosphere* 17(8), 779.**
*Data:* EPA regulatory monitors, PurpleAir sensors and satellite PM2.5, 1998–2023. *Task:* **next-day daily
PM2.5**. *Design:* train 1998–2021, validation 2022, test 2023 — the same year-based design as ours.
*Result:* for the Dallas 2023 hold-out, **XGBoost had the highest R² (0.667)**, ahead of Random Forest
(0.588), Gradient Boosting (0.572) and the fused LSTM (0.555).
*Where to look:* Section 3.6, subsection "Baseline Model Comparisons", **Table 6** ("Baseline model
comparison for Dallas, TX, 2023 temporal holdout"); split years in **Table 4** (Section 3.3) and Section 3.5.
*Meaning for us:* on daily data, tree boosting beat deep learning; supports **XGBoost** and **Random Forest**
and our year-based split.

**[L6] Hyso & Gjylapi (2026), eight Albanian cities — *Baltic Journal of Modern Computing* 14(3), 618–644.**
*Data:* Open-Meteo/CAMS modelled air-quality fields (an open API product, like ours), 2024–2026. *Task:*
**1-, 2- and 3-day-ahead** daily mean PM2.5. *Models:* naive, seasonal naive, drift, Holt, ARIMA, Random
Forest (local baselines); persistence, **Ridge** and HistGradientBoosting with global features; XGBoost as a
challenger. *Design:* rolling-origin backtesting.
*Result:* for 1-day PM2.5 the best enhanced model (HistGBM) had **MAE = 1.92** versus **2.04** for the best
baseline (Random Forest with 7 lags); sMAPE 17.40% vs 18.15%. After Bonferroni adjustment none of the six main
improvements was clearly significant.
*Where to look:* **Table 4, page 631** (MAE comparison); Section 5.2, page 632 (sMAPE and significance);
Section 4.1, pp. 623–624 (rolling-origin design); Section 4.4, pp. 625–626 (Bonferroni analysis); Sections
4.2–4.3 (models); Section 4.7 and Tables 9–10 (XGBoost).
*Meaning for us:* the most similar data product and horizon; gains of ML over strong baselines are real but
small. Supports **Ridge** and **gradient boosting**.

**[L7] Abuouelezz et al. (2025), Abu Dhabi, UAE — *Scientific Reports* 15, 9797.**
*Data:* 6 monitoring stations, hourly, 2018–2022; split 80% training (4 years) / 20% testing (1 year).
*Task:* forecasting at 1 h, 2 h, 1 day and 1 week. *Models:* Decision Tree, Random Forest, linear SVR, RBF
SVR, 1D-CNN, LSTM, Prophet.
*Result:* **linear SVR was best for PM2.5 one hour ahead** (average over stations: MAE = 6.630,
RMSE = 15.911, MAPE = 0.187, i.e. 18.7%).
*Where to look:* **Table 5** (PM2.5 results by horizon and model; best values in bold, last row = average
across stations); "Data preprocessing" subsection of the Methodology for the 80/20 split. The 1-day values
are in the same table (the web version is hard to read; use the PDF).
*Meaning for us:* another study where **SVR** wins; shows the drop in accuracy towards a 1-day horizon.

**[L8] Bhatta, Acharya & Yang (2026), Bangkok, Thailand — *Environmental Challenges* 22, 101442.**
*Data:* 12 stations, 1,827 daily observations, 2020–2024, with meteorology. *Models:* Random Forest,
Gradient Boosting, LSTM; temporal lags, moving averages and cyclical encoding.
*Result:* **Gradient Boosting performed best** (RMSE = 2.17 µg/m³); regularization limited the drop from
training to external validation to 3.9% (GB) and 5.6% (RF).
*Where to look:* **Abstract** (Macquarie University and Mahidol University repository records; full text via
the DOI).
*Meaning for us:* daily PM2.5 with lag and moving-average features; boosting outperformed LSTM. Supports
**XGBoost** and our feature design.

**[L9] Domalatov et al. (2025), Almaty, Kazakhstan — *International Journal of Sustainable Development and Planning* 20(12), 5227–5246.**
*Data:* PM2.5 and PM10 from 71 Almaty stations (AQICN, AirKaz, air.org.kz), 2020–2024, aggregated to daily
city medians; meteorology (temperature, humidity, wind, boundary-layer height, pressure, cloud cover) from ERA5
and Ogimet. *Task:* forecasting 7 and 30 days ahead on a chronological hold-out. *Models:* Random Forest, LSTM;
persistence and seasonal-naive baselines. Two scenarios: A = pollutant lags only, B = pollutant lags **plus
meteorology**.
*Result:* for PM2.5 at the 7-day horizon, **adding weather reduced Random Forest RMSE from 6.22 to 3.96 µg/m³
and MAE from 4.95 to 3.30 µg/m³**; LSTM improved less (RMSE 7.61 → 6.49). **Random Forest was best for
PM2.5** in every PM2.5 setting. The baseline RMSE for PM2.5 at 7 days was 6.40.
*Where to look:* **Table 3** (RMSE, MAE, MAPE for RF and LSTM, scenarios A/B, 7 and 30 days) and **Table 4**
(baseline comparison). Note: Table 4 repeats some Table 3 values with small inconsistencies (e.g. RF B
PM2.5 30-day 6.91 vs 5.70); we cite Table 3.
*Meaning for us:* the most recent study from Kazakhstan and the strongest evidence that **meteorological
inputs** are the main source of improvement for PM2.5 forecasts; supports adding weather features and keeping
Random Forest.

### B. Estimation studies (same-time inputs) — useful for context, not directly comparable

**[L2] Agibayeva et al. (2022), Astana, Kazakhstan — *Sustainability* 14(24), 16641.**
*Data:* two Kazhydromet stations, 2019, daily means. *Inputs:* concurrent PM10, SO2, CO, NO2, NO and weather.
*Models:* multiple linear regression (MLR) and Random Forest; 80/20 split with 10-fold CV.
*Result:* **Random Forest performed better**, with R² between 0.79 and 0.98; MLR gave R² = 0.84 (heating),
0.68 (non-heating) and 0.99 (combined).
*Where to look:* Section 2.4 (split and CV); Section 3.4 and **Table 5** (MLR); Section 3.5 and **Table 6**
(Random Forest); Abstract for the RF range.
*Meaning for us:* the only verified ML study for Astana; motivates the heating-season analysis and the
**Random Forest** choice. Not a forecast (same-day inputs).

**[L10] Haseeb et al. (2026), Lahore, Pakistan — *Atmosphere* 17(4), 411.**
*Data:* satellite (MODIS, Sentinel-5P), ERA5 reanalysis and one ground monitor, 2019–2024; independent
2024 test year. *Models:* RF, XGBoost, LightGBM, CatBoost and a weighted ensemble.
*Result:* among single models Random Forest was best (R² = 0.767, RMSE = 24.75), then XGBoost (R² = 0.76,
RMSE = 26.32), CatBoost (R² = 0.73) and LightGBM (R² = 0.70); a **weighted ensemble of RF, XGBoost and CatBoost
was best overall** (R² = 0.7728, RMSE = 23.37; MAE = 17.29).
*Where to look:* Sections 3.2.1–3.2.5 and 3.3 with **Table 2** ("Independent validation performance … using
the 2024 dataset"); MAE only in Section 5 (Conclusions). In Section 4.3 the authors state the model cannot be
used alone to predict future concentrations, because it relies on concurrent pollutant inputs. LightGBM and
CatBoost values are in the same Table 2 and in the Abstract.
*Meaning for us:* RF/XGBoost beat LightGBM/CatBoost, and combining models helped — the basis for our
**ensemble (Model 5)**; also shows why concurrent inputs must not be used for forecasting.

**[L12] Makhdoomi, Sarkhosh & Ziaei (2025), Mashhad, Iran — *Scientific Reports* 15, 8076.**
*Data:* 2,203 matched observations, 2016–2022; random 80/20 split; inputs include concurrent variables.
*Models:* LightGBM, XGBoost, RF, Gradient Boosting Regression (GBR).
*Result:* GBR R² = 0.9802 (RMSE 2.31), XGBoost R² = 0.9781, RF R² = 0.9686 (MAE 0.47).
*Where to look:* **Table 1** (model comparison) in the full-text PDF.
*Meaning for us:* a cautionary example — near-perfect R² from same-time inputs and a random split is not
comparable with a genuine next-day forecast.

---

## 2.3. Traditional regression methods

Linear models appear as a reference in almost every study. They were competitive in Astana (MLR R² up to
0.99 with concurrent inputs; L2) and were one of the enhanced models in the Albanian day-ahead benchmark
(L6). Their weakness is the inability to model interactions such as "the same recent level means something
different in winter than in summer". **Ridge regression** (Hoerl & Kennard, 1970) adds an L2 penalty that
stabilizes coefficients when many lag and rolling features are correlated — exactly our situation.

## 2.4. Tree-based ensemble models

Tree ensembles were the most frequent winners among forecasting studies: XGBoost in Prishtina (L3) and
Dallas (L5), Gradient Boosting in Bangkok (L8), HistGradientBoosting in Albania (L6); Random Forest was best
in Astana (L2) and the strongest single model in Lahore (L10). **Random Forest** (Breiman, 2001) averages
many decorrelated trees; **XGBoost** (Chen & Guestrin, 2016) adds trees sequentially with regularization. On
daily data both beat LSTM where compared (L5, L8).

## 2.5. Support vector regression

SVR was the most consistently best model for next-day PM2.5 in Greece (L4) and the best PM2.5 model one hour
ahead in Abu Dhabi (L7). Its ε-insensitive loss ignores small errors and the RBF kernel captures smooth
non-linear effects (Smola & Schölkopf, 2004). It needs feature scaling and is efficient for datasets of our
size (~1,800 days).

## 2.6. Time-series forecasting methods

Lags and rolling statistics of PM2.5 were the core features in every forecasting study (L1, L4, L5, L6, L8),
usually with sine/cosine calendar encoding. Feature importance repeatedly shows recent PM2.5 and short
rolling means as dominant (L1: current value ≈0.72 and rolling mean ≈0.24 of Random Forest importance). The
strongest designs fix chronological boundaries before processing (L4), hold out whole years (L5, L7, L10) or
use rolling-origin backtesting (L6), and include a persistence baseline (L1, L4, L6).

## 2.7. Data sources and dataset limitations

| Data type | Studies | Strength | Limitation |
|---|---|---|---|
| OpenWeather API | L1 | free, global, consistent, programmatic | modelled values for a coordinate; production method not documented (OpenWeather, n.d.) |
| Open modelled fields (CAMS/Open-Meteo) | L6 | reproducible, multi-city | model output, not measurements |
| Ground stations | L2, L4, L7, L8, L9, L12 | direct measurements | point coverage, gaps |
| Satellite / reanalysis (ERA5) | L5, L9 (weather), L10 | spatial coverage; complete weather records | coarse; reanalysis is not a forecast |

## 2.8. Comparative analysis

**Table 2.1.** Best model and verified result in each study.

| ID | Study | Task | Horizon | Best model | Reported result | Where to look | Comparable with ours? |
|---|---|---|---|---|---|---|---|
| L1 | Sapakova 2026 (Almaty) | forecasting | 15–60 min | **Persistence** | R² 0.980 vs ML 0.962–0.977 (15 min) | §3.5, Table 7, Fig. 9 | partly (same API, very short horizon) |
| L3 | Fetahi 2025 (Prishtina) | forecasting | 1 h, 24 h | **XGBoost** | R² 0.862, MAE 3.524 (1 h); R² 0.50 (24 h) | Abstract | partly |
| L4 | Psomadakis 2026 (Greece) | forecasting | **next day** | **SVM** | 13/20 wins; all ML > persistence | Abstract, §6.1, Table 8, Fig. 9, §8 | **yes, closest** |
| L5 | Chivukula 2026 (USA) | forecasting | **next day** | **XGBoost** | R² 0.667 (RF 0.588, LSTM 0.555) | §3.6, Table 6 | **yes** |
| L6 | Hyso 2026 (Albania) | forecasting | **1–3 days** | **HistGBM** | MAE 1.92 vs 2.04 baseline | Table 4, p. 631 | **yes** |
| L7 | Abuouelezz 2025 (UAE) | forecasting | 1 h – 1 week | **Linear SVR** (1 h) | MAE 6.630, RMSE 15.911 | Table 5 | partly |
| L8 | Bhatta 2026 (Bangkok) | forecasting | 1–7 days | **Gradient Boosting** | RMSE 2.17 | Abstract | partly |
| L9 | Domalatov 2025 (Almaty) | forecasting | 7 and 30 days | **Random Forest + weather** | RMSE 6.22 → 3.96 with meteorology | Table 3 | partly (daily, Kazakhstan, longer horizon) |
| L2 | Agibayeva 2022 (Astana) | estimation | none | **Random Forest** | R² 0.79–0.98 | §3.5, Table 6 | no |
| L10 | Haseeb 2026 (Lahore) | estimation | none | **Ensemble (RF/XGB/CatBoost)** | R² 0.7728 (RF 0.767 > XGB 0.76 > CatBoost 0.73 > LightGBM 0.70) | §3.3, Table 2 | no |
| L12 | Makhdoomi 2025 (Mashhad) | estimation | none | **GBR** | R² 0.9802 | Table 1 | no |

**What the comparison shows.** In the five studies that are closest to our task (L4, L5, L6, L7, L8 —
future forecasting, daily or near-daily), the winners are **SVM/SVR** (L4, L7) and **gradient-boosted
trees** (L5, L6, L8). Random Forest is consistently close behind and was best in the Astana estimation study
(L2). Linear models are weaker but remain a meaningful reference (L6). LSTM never won on daily data (L5, L8).
Persistence is strong at very short horizons (L1) and must always be reported (L4, L6). The very high scores
of L2, L10 and L12 come from estimation designs and are not comparable with ours. Finally, the Almaty study
by Domalatov et al. (L9) shows that the largest single improvement comes not from changing the algorithm but from
**adding meteorological information** (−36% RMSE for Random Forest).

## 2.9. Strengths and limitations of the literature

* Short-horizon results (L1, L3, L7) overstate what is possible for a day-ahead daily mean.
* Concurrent inputs (L2, L10, L12) and random splits (L2, L12) inflate accuracy.
* Reporting is inconsistent: some studies give only R² (L4, L5), others only MAE/sMAPE (L6).
* Central Asian evidence is limited to Almaty (L1 nowcasting, L9 7–30-day forecasting) and Astana estimation (L2); none forecasts the next-day mean for Astana.
* Modelled API data (L1, L6) were not validated against ground stations in those studies.

## 2.10. Model selection derived from the literature

**Table 2.2.** From evidence to the models we train.

| Model family | Evidence in similar studies | Decision |
|---|---|---|
| **Support Vector Regression (RBF)** | best next-day model in Greece (L4: 13/20 wins); best 1-h model in UAE (L7) | **Selected** (Model 4) |
| **Gradient-boosted trees** | best in Dallas next-day (L5: XGBoost R² 0.667), Albania day-ahead (L6: HistGBM), Bangkok (L8), Prishtina (L3) | **Selected as XGBoost** (Model 3) |
| **Random Forest** | best in Almaty with weather (L9), Astana (L2), best single model in L10, second in L5 | **Selected** (Model 2) |
| **Regularized linear regression** | competitive reference in L6 (Ridge) and L2 (MLR) | **Selected as Ridge** (Model 1) — interpretable baseline model |
| **Persistence** | beats ML at 15 min in Almaty (L1); minimum bar in L4 and L6 | **Selected as non-ML baseline** |
| LSTM / deep learning | never best on daily data (L5: R² 0.555 < XGBoost 0.667; L8); needs more data than our ~1,800 days | **Not selected** |
| SARIMA | worse than persistence at most stations (L4) | **Not selected** |
| **Ensemble of the best models** | weighted RF+XGBoost+CatBoost ensemble best in L10 | **Selected as Model 5**: equal-weight average of tuned RF + XGBoost + SVR |
| **Meteorological inputs** | RF RMSE −36% with weather (L9); weather drives episodes (L8, L10) | **Added**: Open-Meteo past weather + target-day weather (configs `full_weather`, `full_weather_past`) |
| LightGBM / CatBoost | below RF and XGBoost in L10 (R² 0.70 / 0.73); HistGBM competitive in L6 | **Not selected** — same family as XGBoost, no evidence of a gain |

**Feature and evaluation decisions taken from the same studies:**

* lags, rolling means/SD and cyclical calendar features (L1, L4, L5, L6, L8);
* only information available before the forecast time — no concurrent co-pollutants (lesson from L10, L12);
* year-based split: train / validation year / independent test year (L5, L7, L10);
* persistence baseline on identical dates (L1, L4, L6);
* separate evaluation of high-pollution days and the heating season (L2, L10);
* MAE as the primary metric plus RMSE, R² and sMAPE, because studies report different metrics (L4, L6);
* meteorological features from a reanalysis archive for training and a weather forecast for live use (L9).

## 2.11. Research gap

The verified Kazakh studies address same-day estimation in Astana with station data (L2), 15-minute
nowcasting in Almaty with OpenWeather data (L1), or 7–30-day forecasting in Almaty with station and ERA5 data
(L9). We did not find a study that evaluates a
**day-ahead daily-mean** PM2.5 forecast for **Astana** using **multi-year OpenWeather historical data**,
leakage-free issue-time features, a year-based train/validation/test design and an explicit persistence
baseline. This project addresses that gap; we do not claim that no similar work exists elsewhere.
