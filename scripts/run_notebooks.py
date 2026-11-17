"""Execute all notebooks in place so that they contain real outputs.

Run AFTER the pipeline has produced data and artifacts:
    python scripts/run_notebooks.py
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = sorted((ROOT / "notebooks").glob("0*.ipynb"))

if __name__ == "__main__":
    failed = []
    for p in NB:
        print(f"Executing {p.name} ...", flush=True)
        nb = nbformat.read(p, as_version=4)
        try:
            ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(p.parent)}})
        except Exception as exc:  # keep partial outputs for inspection
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            failed.append(p.name)
        nbformat.write(nb, p)
    print("Done." if not failed else f"Failed notebooks: {failed}")
    sys.exit(1 if failed else 0)
