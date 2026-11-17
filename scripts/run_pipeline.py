"""Run the complete pipeline (equivalent to `python -m air_quality.cli run-all`).

Usage:  python scripts/run_pipeline.py [--quick] [--skip-download] [--skip-training]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_quality.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run-all", *sys.argv[1:]]))
