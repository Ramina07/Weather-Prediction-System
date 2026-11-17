"""API smoke test (current endpoint + 2025-06-01..02 history). Consumes 2 API calls.

Usage:  python scripts/test_api.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_quality.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["test-api"]))
