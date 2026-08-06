"""
Reproducible build: runs the whole data/model pipeline end to end so the
artifacts the app serves can be regenerated from the raw Excel with one command.

    python build_all.py

Order matters: each step consumes the previous step's output.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("pipeline.py",      "clean raw Excel -> data/tests_clean.csv"),
    ("build_lexicon.py", "feature-category doc -> artifacts/lexicon.json"),
    ("classify.py",      "classify test names -> data/tests_classified.csv"),
    ("model.py",         "train P(success) -> artifacts/prediction_table.csv"),
    ("value.py",         "value term -> artifacts/value_table.json"),
    ("score.py",         "assemble scores -> artifacts/scoring_table.csv"),
]


def main():
    for script, desc in STEPS:
        print(f"\n{'='*70}\n>>> {script}  —  {desc}\n{'='*70}")
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if r.returncode != 0:
            print(f"FAILED at {script}")
            sys.exit(r.returncode)
    print("\nAll steps completed. Artifacts are in ./artifacts and ./data.")


if __name__ == "__main__":
    main()
