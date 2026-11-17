# Data provenance

| Item | Value |
|---|---|
| Provider | OpenWeather — Air Pollution API (https://openweathermap.org/api/air-pollution) |
| Endpoint used for the dataset | `GET https://api.openweathermap.org/data/2.5/air_pollution/history?lat=&lon=&start=&end=&appid=` |
| Other endpoints | `/air_pollution` (current; smoke test), `/air_pollution/forecast` (provider forecast; optional, labelled separately) |
| Documented availability | Historical data from 27 November 2020; forecast for 4 days hourly (OpenWeather, n.d.) |
| Location | Astana, Kazakhstan; requested coordinate 51.1694 N, 71.4491 E (exact API-returned coordinate saved in `data/raw/dataset_metadata.json`) |
| Timezone | Asia/Almaty (UTC+6 until 2024-02-29; UTC+5 from 2024-03-01, single national zone) |
| Requested period | local 2020-12-26 (lead-in) to 2025-12-31 |
| Units | µg/m³ for CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3; AQI 1–5 (provider scale) |
| Timestamp | `dt`, Unix seconds UTC, start of the hour |
| Acquisition log | `data/raw/chunks/manifest.json` (per-chunk status, record counts, coverage, errors) |
| Raw responses | `data/raw/chunks/chunk_<start>_<end>.json` (verbatim JSON, preserved) |
| Metadata | `data/raw/dataset_metadata.json` (request count, record count, first/last timestamp) |
| Credentials | Read from `.env` (`OPENWEATHER_API_KEY`); redacted from all logs and errors; never stored in outputs |

## Credibility and limitations of the data product

* The API returns concentrations "for any coordinates on the globe"; the documentation page does not describe
  how they are produced (OpenWeather, n.d.). They should be described as **API-provided (modelled/aggregated)
  concentration data**, not as measurements from a particular sensor.
* A single coordinate is not the same as the average over the whole city; intra-urban variability (e.g.
  private-house heating districts) is not represented.
* No independent ground-station validation was performed in this project. Comparison with Kazhydromet
  monitors (as used by Agibayeva et al., 2022) is recommended future work.
* The provider may revise or back-fill historical values; the download date is recorded in the manifest so
  that the exact data snapshot is traceable.
* The model learns to forecast the API's own series. Its accuracy statements apply to that series only.

## Actual coverage

The actual record count, coverage, gaps and missingness are produced by the code after download and are not
written here by hand:

* `python -m air_quality.cli audit` → `artifacts/metrics/dataset_audit.json`
* `artifacts/metrics/data_quality_report.md` (before/after cleaning)
* Notebook `01_dataset_audit.ipynb`

## Meteorological data (added 17 Nov 2026)

| Item | Value |
|---|---|
| Provider | Open-Meteo (Zippenfenig, 2023; https://doi.org/10.5281/zenodo.7970649), no API key |
| Historical endpoint | `https://archive-api.open-meteo.com/v1/archive` (ERA5-based reanalysis "best match") |
| Live endpoint | `https://api.open-meteo.com/v1/forecast` (past_days=5, forecast_days=3) |
| Variables | temperature_2m (°C), relative_humidity_2m (%), surface_pressure (hPa), precipitation (mm), cloud_cover (%), wind_speed_10m (m/s), wind_direction_10m (°) |
| Files | `data/raw/weather/openmeteo_<start>_<end>.json` (raw, per year), `data/raw/weather_openmeteo_hourly.csv` |
| Caveat | In training, target-day weather comes from reanalysis (perfect prognosis); live forecasts use a weather forecast, so real-time accuracy of `full_weather` can be lower than the benchmark. `full_weather_past` uses only weather observed before 18:00. |
