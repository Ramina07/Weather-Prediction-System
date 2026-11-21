# Limitations

1. **Data product.** OpenWeatherMap values are API-provided concentrations for a coordinate; the production
   method is not documented on the API page. They are not ground measurements, and this project did not validate
   them against Kazhydromet stations. All accuracy statements refer to the API series.
2. **Spatial representativeness.** One coordinate cannot represent intra-urban variability (e.g. districts with
   private coal heating vs central areas).
3. **Weather inputs are a forecast.** Version 2 adds Open-Meteo weather. In the historical benchmark the
   target-day weather comes from reanalysis ("perfect prognosis"), which is optimistic; live forecasts use the
   real Open-Meteo forecast, whose own errors add to the model error. `full_weather_past` (no target-day
   weather) is reported as the strictly operational benchmark.
4. **Single test year.** 2025 is one realisation of inter-annual variability; a different year could rank the
   models differently. Rolling-origin evaluation is future work.
5. **Elevated-pollution sample size.** The number of days above the training 90th percentile in 2025 is limited;
   E5 metrics have wide uncertainty.
6. **Fixed issue time.** The model is trained for 18:00 issues only. Earlier forecasts require the explicit
   `--issue-mode latest` protocol (target = today) or retraining for another issue hour.
7. **Operational latency.** If the history endpoint lags behind real time, live forecasts may be refused as
   stale (configurable threshold `forecast.max_staleness_hours`).
8. **Provider revisions.** Historical values may be revised by the provider; the manifest records download dates.
9. **Hyperparameter search budget.** RandomizedSearchCV with 25 candidates per model/configuration is a practical
   compromise, not an exhaustive optimisation.
10. **Interpretability.** Permutation and tree importances are associational and can be diluted among correlated
    features; they do not establish causes.
11. **Prediction interval.** The empirical interval assumes that future residuals resemble 2024 validation
    residuals; its coverage is not guaranteed.
12. **Prototype application.** The Streamlit web application (`app/`) runs locally on one computer and keeps its
    forecast history in a local SQLite file; it is not a deployed multi-user service.
