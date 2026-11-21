# Next-Day PM2.5 Forecasting for Astana (OpenWeatherMap Air Pollution API)

**Development of an Interactive Machine Learning-Based System for Next-Day PM2.5 Air Pollution Forecasting
Using OpenWeatherMap API** — Machine Learning Algorithms (AMO 3208) final project, backend phase.

The system forecasts the **mean PM2.5 (µg/m³) of the next local calendar day** in Astana, Kazakhstan. A forecast
is issued at **18:00 Asia/Almaty** and uses only data available at that moment. Four regression models
(Ridge, Random Forest, XGBoost, SVR) are tuned with time-series cross-validation, compared with a persistence
baseline on a chronological split (train 2021–2023, validation 2024, test 2025), and the selected model is
exposed through a command-line forecaster. The selected model is used by the CLI and by the interactive web application in `app/` (section 5a).

## 1. Setup

**macOS / Linux (one command):** `bash setup_and_run.sh` — creates `.venv`, installs, tests, runs the full pipeline, executes notebooks and prints a forecast. On macOS, XGBoost may need `brew install libomp`.

**Windows PowerShell:** `powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1`, or step by step:

Requires Python 3.10–3.13.

```powershell
cd ML_final
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                       # makes `python -m air_quality.cli` work from anywhere
# optional: pip install -r requirements-optional.txt   (SHAP)
python -m air_quality.cli env          # records versions in artifacts/logs/environment.json
```

## 2. API key

Copy `.env.example` to `.env` and put your key on the line `OPENWEATHER_API_KEY=...`. `.env` is git-ignored;
the key is never printed, logged or saved in outputs. New keys can take up to ~2 h to activate (HTTP 401 meanwhile).

## 3. Run the pipeline

```powershell
python -m air_quality.cli test-api                 # smoke test: current + 2025-06-01..02 history (2 calls)
python -m air_quality.cli download                 # 2021-01-01..2025-12-31 (+6-day lead-in), resumable
python -m air_quality.cli audit                    # coverage, duplicates, missingness, PM2.5 range
python -m air_quality.cli preprocess               # cleaning + daily targets + features
python -m air_quality.cli eda                      # figures 1-7 + EDA statistics
python -m air_quality.cli train                    # 4 models x 5 feature configs, tuned + default, frozen test
python -m air_quality.cli evaluate                 # experiments E1-E5, E7 + figures 8-15
python -m air_quality.cli explain                  # experiment E6 (permutation importance, SHAP if installed)
python -m air_quality.cli report                   # docs/research_report_draft.md filled with real results
# or everything at once:
python -m air_quality.cli run-all
```

`train` runs ~1,000 model fits; expect roughly 10–40 minutes on a laptop. `train --quick` uses a small search
budget for a fast check (not for final results).

## 4. Tomorrow's forecast

```powershell
python -m air_quality.cli predict                        # after 18:00 local: forecast for tomorrow
python -m air_quality.cli predict --issue-mode latest    # before 18:00: forecast issued yesterday 18:00 for TODAY
python -m air_quality.cli predict --with-provider         # also shows the OpenWeatherMap Provider Forecast (separately labelled)
python -m air_quality.cli update                         # download 2026 data for operational use
python -m air_quality.cli train-operational              # refit selected config on all labelled data (stored separately)
```

Output: terminal summary + `artifacts/predictions/latest_forecast.json`, `forecast_<date>_<stamp>.json` and
`forecast_history.csv`. Stale or incomplete live data cause an explicit error (exit 5) unless `--force` is
given, in which case the forecast is flagged `reliable: false`.

## 5. Notebooks and tests

```powershell
python scripts/run_notebooks.py      # executes notebooks 01-05 in place (after the pipeline)
python -m pytest                     # unit, leakage, model and CLI tests (mocked API, synthetic fixtures)
```

Synthetic fixtures exist only inside `tests/` and are never used as experimental data.

## 5a. Web application

An interactive Streamlit application (`app/`) sits on top of the backend:

```bash
pip install -r requirements.txt      # includes streamlit and plotly
streamlit run app/app.py             # opens http://localhost:8501
```

| Page | What it shows |
|---|---|
| Home | project summary, key results, latest forecast |
| Live Forecast | runs `prediction.make_forecast` (real API + Open-Meteo), category, 90 % band, provider comparison, map |
| Track Record | SQLite store of every forecast; real outcomes downloaded automatically and scored; hindcasts |
| Model Arena | leaderboard, accuracy plots, model × feature-set heatmap, tuning results, model guide |
| Time Machine | replay any 2025 test day: each model's forecast vs the real value, hourly profile |
| What-if Simulator | change recent pollution / weather and watch every saved model respond; sensitivity curves |
| Explainability | permutation and tree importance, Ridge coefficients, feature families |
| Data Explorer | interactive EDA of both datasets, data quality, report figures, raw tables |
| Experiments E1–E7 | hypotheses H1–H5 with verdicts computed from `artifacts/metrics/` |
| Methods & Literature | methodology, LaTeX formulas, literature matrix, references, limitations |

Forecasts are stored in `data/app/forecast_store.db` (tables `forecasts`, `observations`, `events`).
The API key is read from `.env` exactly as for the CLI. `scikit-learn` is pinned to 1.6.1, the version the
saved models were trained with.

## 6. Project layout

```
config.yaml                 all parameters (location, periods, protocol, splits, search budget)
src/air_quality/            package: config, api_client, download, data_validation, preprocessing,
                            feature_engineering, feature_selection, models, training, evaluation,
                            explainability, eda, prediction, report, cli
notebooks/01..05            audit, EDA, preprocessing & features, training, evaluation
scripts/                    run_pipeline.py, test_api.py, run_notebooks.py, build_notebooks.py, build_literature_table.py
tests/                      pytest suite
data/raw|interim|processed  raw API data (+ chunk cache/manifest), clean hourly grid, feature table
artifacts/                  models/{benchmark,operational}, metrics, figures, predictions, logs
docs/                       literature review & matrix, references, methodology, formulas, data docs,
                            report draft, limitations, defense notes, requirements checklist
```

**Which code corresponds to each model:** `src/air_quality/models.py` — `make_estimator("ridge" | "random_forest" |
"xgboost" | "svr")`, `param_space(...)`, `PersistenceForecaster`; training loop in `training.run_training`.

## 7. Exit codes

0 OK · 1 unexpected · 2 config · 3 missing API key · 4 API error · 5 data/stale data · 6 missing model · 7 issue-time protocol.

## 8. Data notice

Values are API-provided (modelled/aggregated) concentrations for coordinate 51.1694 N, 71.4491 E, not
ground-station measurements, and they were not validated against stations in this project.
