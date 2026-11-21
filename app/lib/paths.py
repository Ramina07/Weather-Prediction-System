"""Project paths. Makes the backend package (``src/air_quality``) importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# the backend looks for config.yaml from the working directory or this variable
os.environ.setdefault("AIR_QUALITY_ROOT", str(ROOT))

from . import compat  # noqa: E402,F401  (scikit-learn pickle compatibility)

ASSETS = APP_DIR / "assets"
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
METRICS = ARTIFACTS / "metrics"
FIGURES = ARTIFACTS / "figures"
PREDICTIONS = ARTIFACTS / "predictions"
MODELS = ARTIFACTS / "models"
DOCS = ROOT / "docs"
STORE_DB = DATA / "app" / "forecast_store.db"
