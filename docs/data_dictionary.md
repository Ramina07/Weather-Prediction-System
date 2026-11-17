# Data dictionary

## Raw hourly table — `data/raw/openweather_astana_2021_2025.csv`

| Column | Type | Unit | Description |
|---|---|---|---|
| dt_unix | int64 | s | OpenWeatherMap `dt`, Unix time (UTC) of the **start** of the hour |
| timestamp_utc | string (ISO 8601) | – | `dt_unix` as UTC timestamp |
| timestamp_local | string (ISO 8601 with offset) | – | Asia/Almaty local time |
| date_local | string (YYYY-MM-DD) | – | Local calendar date |
| latitude, longitude | float | degrees | Coordinates returned by the API (requested 51.1694, 71.4491) |
| provider | string | – | "OpenWeatherMap Air Pollution API (history)" |
| aqi | float | index 1–5 | `main.aqi` (1 Good … 5 Very Poor), provider scale |
| co | float | µg/m³ | carbon monoxide (`components.co`) |
| no | float | µg/m³ | nitrogen monoxide |
| no2 | float | µg/m³ | nitrogen dioxide |
| o3 | float | µg/m³ | ozone |
| so2 | float | µg/m³ | sulphur dioxide |
| pm2_5 | float | µg/m³ | fine particulate matter (target pollutant) |
| pm10 | float | µg/m³ | coarse particulate matter |
| nh3 | float | µg/m³ | ammonia |

## Clean hourly grid — `data/interim/hourly_air_pollution_clean.csv`

Raw value columns plus: `hour_local` (int), `is_observed` (bool; False = missing hour inserted as NaN),
`pm25_extreme_flag` (bool; robust z of log1p(PM2.5) > 3.5, retained).

## Feature dataset — `data/processed/daily_pm25_features.csv` (one row per issue date)

| Column | Description |
|---|---|
| issue_date, issue_time_local, issue_time_utc | Forecast issue: day *d* at 18:00 Asia/Almaty |
| target_date | *d*+1 |
| target_pm25 | Mean PM2.5 of target_date (NaN if < 20 valid hours) |
| target_valid_hours, target_is_valid | Coverage of the target day |
| meta_pm25_coverage_72h | Share of valid PM2.5 hours in the 72 h before the issue |
| meta_last_obs_age_hours | Age (h) of `pm25_last` relative to the 17:00 record |
| meta_newest_record_utc | Newest grid record that could be read at the issue time |
| eligible | Valid target and persistence inputs available (used for all models) |
| split | train / validation / test / unused (by target date) |
| target_dow … target_is_heating_season | Calendar features of the target day (9) |
| pm25_last | Most recent PM2.5 (≤ 3 h old), µg/m³ |
| pm25_lag_24h/48h/72h | PM2.5 24/48/72 h before the 17:00 record |
| {pm10,no2,co,o3,so2,nh3,no}_last | Most recent value of each pollutant |
| {pm10,no2,co,o3}_lag_24h | 24-h lags |
| pm25_mean_{6,12,24,48,72}h | Trailing means ending at 17:00 (coverage ≥ 75%) |
| pm25_std_24h, pm25_min_24h, pm25_max_24h | Trailing 24-h dispersion and range |
| pm25_trend_24h | OLS slope over trailing 24 h (µg/m³ per h) |
| pm25_diff_24h | mean(last 24 h) − mean(previous 24 h) |
| pm25_prev_day_mean | Mean of local day *d*−1 |
| pm_ratio_24h | pm25_mean_24h / pm10_mean_24h |
| {pollutant}_mean_24h | Trailing 24-h mean for PM10, NO2, CO, O3, SO2, NH3, NO |
| aqi_last, aqi_mean_24h | Historical AQI (ablation configuration only) |

## Prediction files — `artifacts/predictions/`

`predictions_validation.csv`, `predictions_test.csv`: issue_time_local, target_date, actual, predicted,
abs_error, model, feature_config, variant (tuned/default/baseline), stage.
`latest_forecast.json`, `forecast_<date>_<stamp>.json`, `forecast_history.csv`: live forecasts.
