"""
Phase 3 / Tier 1 - Value term.

'Value' = expected first-year profit IF the test wins. Tier 1 switches the basis
from actual winner profit (winners only, 82 rows, survivorship-biased upward) to
the pre-launch **expected_profit** estimate, which:
  * exists for losers too (113) as well as winners (27) -> no survivorship bias,
  * covers 140 vs 82 tests,
  * correlates r=0.76 with realized profit among winners.

Estimated at the placement level with a minimum-support fallback to the global
median. At serve time a PM's own expected-profit input overrides this (handled in
score.py); this table is the fallback.

PRIVACY (public deploy): we publish only a RELATIVE index (placement median /
global median) — no absolute $ figure is ever written to disk or shown, unless the
PM enters their own profit number (which is their data, not the company's).
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tests_classified.csv"
OUT = HERE / "artifacts" / "value_table.json"

MIN_SUPPORT = 5


def main():
    df = pd.read_csv(DATA)
    tr = df[df["in_training"]].copy()
    w = tr[tr["expected_profit"].notna()].copy()

    global_median = float(w["expected_profit"].median())
    # PRIVACY: we publish only the RELATIVE value index (placement median / global
    # median), never absolute $ figures. The global median is used here to compute
    # ratios and is then discarded — no company revenue number is committed.
    by_place = {}
    for pl, grp in w.groupby("placement_clean"):
        med = float(grp["expected_profit"].median())
        by_place[str(pl)] = {"ratio": round(med / global_median, 3), "n": int(len(grp))}

    out = {
        "basis": "relative value index = placement median expected_profit / global "
                 "median (absolute $ withheld for privacy; global ratio = 1.0)",
        "n_total": int(len(w)),
        "min_support": MIN_SUPPORT,
        "by_placement": by_place,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"basis: relative index (global median withheld) | n={len(w)}")
    for k, v in sorted(by_place.items(), key=lambda kv: -kv[1]["ratio"]):
        tag = "own" if v["n"] >= MIN_SUPPORT else "-> global (1.0) fallback"
        print(f"  {k:<12} n={v['n']:<3} ratio={v['ratio']:>5.2f}x typical   {tag}")
    print(f"\nWrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
