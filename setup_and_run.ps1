# One-shot setup and full pipeline for Windows PowerShell.
# Usage (from the ML_final folder):   powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host ">>> Creating virtual environment (.venv)"
    py -3 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .

$py = ".\.venv\Scripts\python.exe"
& $py -m air_quality.cli env
& $py -m pytest
& $py -m air_quality.cli run-all
if ($LASTEXITCODE -ne 0) { Write-Host "Pipeline stopped with exit code $LASTEXITCODE"; exit $LASTEXITCODE }
& $py scripts\run_notebooks.py
& $py -m air_quality.cli predict --issue-mode latest --with-provider
Write-Host "`nDone. Results: artifacts\metrics, artifacts\figures, docs\research_report_draft.md"
