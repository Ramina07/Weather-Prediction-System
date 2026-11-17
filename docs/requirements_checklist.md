# Requirements-compliance checklist

Status legend: **COMPLETED AND VERIFIED** · **IMPLEMENTED BUT NOT EXECUTED** (code/tests exist; needs real data
run) · **BLOCKED / NOT YET IMPLEMENTED**. Status as of 2026-11-13 (build environment had no access to the API).

## Course criteria (Final Project Structure & Methodical Guidelines)

| Requirement | Where | Status |
|---|---|---|
| 3-person team: 3–5 ML models | 4 models + persistence (`models.py`) | IMPLEMENTED BUT NOT EXECUTED on real data (verified on synthetic tests; XGBoost untested in build env) |
| Application demonstrating practical use | CLI `predict` | IMPLEMENTED BUT NOT EXECUTED; **GUI not implemented in this phase** — confirm with instructor whether CLI is acceptable |
| Abstract, Introduction (25%), Literature Review, Methods (10%), Results & Discussion (20%), Conclusion (15%), References | `docs/research_report_draft.md` (generated) | Text COMPLETED; numerical results PENDING |
| 10–12 sources from last 3–5 years | 12 studies 2022–2026 | COMPLETED AND VERIFIED (metadata; some tables partially accessible — see matrix) |
| 12–18 APA references | 18 in `references_apa.md` | COMPLETED AND VERIFIED |
| Dataset source, size, structure, variable types | `data_provenance.md`, `data_dictionary.md`, notebook 01 | Documented; actual size PENDING download |
| Cleaning, normalization, feature selection | `preprocessing.py`, pipelines, `feature_selection.py` | IMPLEMENTED; tested |
| Training / validation / testing | `training.py` | IMPLEMENTED; tested on synthetic data |
| Mathematical formulations typed (no screenshots) | `mathematical_formulations.md` (LaTeX) | COMPLETED |
| ≥ 3 preprocessing/ML methods compared | basic/lag/full/full_selected × 4 models | IMPLEMENTED BUT NOT EXECUTED |
| Tables and figures labelled | figures 1–15 with titles/units | IMPLEMENTED BUT NOT EXECUTED |
| Numbered conclusions with numerical values | Report §5 (RO1–RO8) | Template COMPLETED; numbers PENDING |
| Report in Word **and** PDF | – | NOT YET DONE (after real results) |
| Presentation (slides) | – | NOT YET DONE (after real results) |
| Source code with model mapping | `README.md` §6, `models.py` | COMPLETED |
| Each member explains contribution | `defense_preparation.md` §1 | Proposed plan; team must confirm |

## Master-prompt quality gates

| Gate | Item | Status |
|---|---|---|
| Dataset | Real API data obtained, period, counts, coverage, missingness, duplicates, units | BLOCKED in build env → run `download` + `audit` |
| Preprocessing | Cleaning, target, no leakage, normalization & selection fitted correctly, feature matrix documented | COMPLETED AND VERIFIED by tests (synthetic) |
| Model | Ridge, RF, XGBoost, SVR trained; persistence; same target; validation; independent test; saved | IMPLEMENTED BUT NOT EXECUTED on real data |
| Research | 10–12 verified publications, matrix, metrics transcribed, gap, model justification | COMPLETED AND VERIFIED (with documented access limits) |
| Practical | Saved model loads, CLI predicts, latest data, tomorrow's date, output saved, no secret exposure | Verified with mocked API + synthetic model; real run PENDING |
| Report | Intro, review, methods, formulas, real results, discussion, conclusions, APA, limitations, reproducibility | Real results PENDING |
| Testing | Tests run, failures corrected, environment documented | COMPLETED (70 passed, 2 skipped: xgboost absent in build env) |

## Security

| Item | Status |
|---|---|
| `.env` git-ignored, `.env.example` placeholder only | COMPLETED |
| Key redacted from logs/errors (`config.redact`, tested) | COMPLETED AND VERIFIED |
| Key shared in chat during setup | Recommended: regenerate the key in the OpenWeather account after the project is finished |
