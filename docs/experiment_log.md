# Experiment log

Chronological record of what was actually executed. Entries are added by the team; automatically generated
metrics live in `artifacts/metrics/`.

| Date (local) | Step | Command | Result / status | Notes |
|---|---|---|---|---|
| 2026-11-13 | Project scaffold, package, tests, notebooks, docs | – | COMPLETED | Built in a cloud workspace without access to `api.openweathermap.org` (network policy) |
| 2026-11-13 | Unit/leakage/model/CLI tests | `python -m pytest` | 70 passed, 2 skipped (xgboost not installed in the build environment) | Synthetic fixtures only |
| 2026-11-13 | End-to-end smoke run on synthetic data (Ridge, RF, SVR) | `run-all` equivalent in a temporary folder | COMPLETED | Verifies code paths only; synthetic outputs were deleted and are not results |
| 2026-11-13 | Literature verification | web verification of 14 candidates | 12 included, 2 excluded (unretrievable), 2 screened out | See `literature_comparison.md` |
| 2026-11-17 | API smoke test | `test-api` | PASSED | MacBook Air (arm64), Python 3.9.6 |
| 2026-11-17 | Historical download | `download` | 43,273 hourly records, 98.4% coverage | |
| 2026-11-17 | Audit, preprocessing, EDA | `audit`, `preprocess`, `eda` | COMPLETED | 1,061/351/346 samples |
| 2026-11-17 | Training (E1–E4) | `train` | COMPLETED (~8 min) | Selected: SVR, full_selected |
| 2026-11-17 | Evaluation, interpretability | `evaluate`, `explain` | COMPLETED | SHAP skipped (optional) |
| 2026-11-17 | Live forecast | `predict --issue-mode latest` | 2.53 µg/m³ for 2026-11-17 | |
| 2026-11-17 | Notebook execution | `run_notebooks.py` | COMPLETED (5/5) | |

## Experiment definitions

For hypotheses, subsets, features, models and evaluation methods of E1–E7 see
`docs/research_methodology.md` §9. Actual results and their interpretation are written into
`docs/research_report_draft.md` by `python -m air_quality.cli report`.
