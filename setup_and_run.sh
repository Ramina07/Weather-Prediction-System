#!/usr/bin/env bash
# One-shot setup and full pipeline for macOS / Linux.
# Usage (from the ML_final folder):   bash setup_and_run.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY -m pip install -e .
$PY -m air_quality.cli env
$PY -m pytest
$PY -m air_quality.cli run-all
$PY scripts/run_notebooks.py
$PY -m air_quality.cli predict --issue-mode latest --with-provider
echo "Done. Results: artifacts/metrics, artifacts/figures, docs/research_report_draft.md"
