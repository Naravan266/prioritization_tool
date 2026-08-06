"""
Phase 0 - Data pipeline for the A/B test prioritization tool.

Reads the raw experiment export and produces one clean, documented table
(tests_clean.parquet) that every downstream stage builds on. Also prints a
"data map" so we can see, honestly, how much of each field is actually filled.

Design decisions (see writeup):
  * SUCCESS LABEL (closed-world, user-confirmed):
        is_success = (isSuccessful == 'Yes')  OR  (firstYearProfitUplift present)
        everything else (blank / 'Partially' / 'Degrading testing') = not success.
  * TRAINING POPULATION: finished, non-pre-test experiments only.
        'active' (outcome unknown) and 'pre_test' (feasibility pilots) are flagged
        and EXCLUDED from model training - they are not failures.
  * PLACEMENT has whitespace-duplicate values (e.g. 'Sitewide' vs 'Sitewide ')
    that must be collapsed.
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "ab_tests_raw.xlsx"
OUT = HERE / "data" / "tests_clean.csv"


# --------------------------------------------------------------------------
# KPI family mapping: collapse ~24 noisy KPI strings into a handful of families.
# Order matters - first matching pattern wins. 'unsubscribe' is a churn/negative
# KPI (direction is inverted) and is kept separate on purpose.
# --------------------------------------------------------------------------
KPI_FAMILY_RULES = [
    ("unsubscribe",   r"unsubscribe"),
    ("add_to_queue",  r"add to q"),                       # covers 'qeue' typo
    ("upgrade_addon", r"upgrade|add-?on|upchage"),
    ("recurring",     r"recurring"),
    ("case_sub",      r"case subscription"),
    ("ecommerce",     r"ecommerce"),
    ("opt_in",        r"opt-?in"),
    ("subscription",  r"subscription|drift subscription|leads"),
    ("account",       r"card update|card"),
]


def map_kpi_family(kpi: object) -> str:
    if not isinstance(kpi, str) or not kpi.strip():
        return "unknown"
    low = kpi.lower()
    for family, pat in KPI_FAMILY_RULES:
        if re.search(pat, low):
            return family
    return "other"


def coerce_money(series: pd.Series) -> pd.Series:
    """Coerce a profit column to numeric dollars; non-numeric junk -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def normalize_placement(series: pd.Series) -> pd.Series:
    """Trim whitespace so 'Sitewide ' and 'Sitewide' collapse; keep NaN as NaN."""
    return series.astype("string").str.strip().replace({"": pd.NA})


def build() -> pd.DataFrame:
    df = pd.read_excel(RAW)
    n0 = len(df)

    # --- clean placement (whitespace-duplicate collapse) ---
    df["placement_clean"] = normalize_placement(df["Placement"])

    # --- coerce profit fields ---
    df["expected_profit"] = coerce_money(df["Expected profit uplift"])   # pre-launch estimate
    df["actual_profit"] = coerce_money(df["firstYearProfitUplift"])       # realized, winners only

    # --- KPI family ---
    df["kpi_family"] = df["KPI name"].apply(map_kpi_family)

    # --- population flags ---
    df["is_active"] = df["Status"].astype("string").str.lower().eq("active")
    df["is_pretest"] = df["Is Pre-Test"].astype(bool)
    # eligible for training: finished, real (non-pilot) experiments with a knowable outcome
    df["in_training"] = (~df["is_active"]) & (~df["is_pretest"])

    # --- SUCCESS LABEL (closed-world) ---
    # NOTE: blank isSuccessful must resolve to False, NOT to pandas <NA>.
    # .eq() on a nullable "string" column yields <NA> for blanks, which would
    # silently drop those rows from means/negatives - so we fillna(False).
    yes = (df["isSuccessful"].astype("string").str.strip().str.lower()
           .eq("yes").fillna(False).astype(bool))
    has_profit = df["actual_profit"].notna()
    df["is_success"] = (yes | has_profit).astype(bool)
    # For non-training rows the label is undefined (active) / not-comparable (pilot);
    # keep the boolean but downstream training must filter on in_training.

    # --- derived: duration ---
    df["duration_days"] = (df["End date"] - df["Start date"]).dt.days
    df["start_year"] = df["Start date"].dt.year

    assert len(df) == n0
    return df


def data_map(df: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"ROWS: {len(df)}")
    print("-" * 70)
    print("POPULATION")
    print(f"  finished, non-pretest (training pool): {df['in_training'].sum()}")
    print(f"  active (excluded, outcome unknown):    {df['is_active'].sum()}")
    print(f"  pre-test pilots (excluded):            {df['is_pretest'].sum()}")
    print("-" * 70)
    tr = df[df["in_training"]]
    print("SUCCESS LABEL (training pool only)")
    print(f"  success = 1: {tr['is_success'].sum()}  "
          f"({tr['is_success'].mean()*100:.1f}% base rate)")
    print(f"  success = 0: {(~tr['is_success']).sum()}")
    print("-" * 70)
    print("FIELD FILL (training pool)")
    for c, label in [("placement_clean", "placement"),
                     ("kpi_family", "kpi_family (non-unknown)"),
                     ("Expected relative uplift", "expected_rel_uplift"),
                     ("expected_profit", "expected_profit"),
                     ("actual_profit", "actual_profit"),
                     ("Stage", "stage")]:
        if c == "kpi_family":
            filled = (tr[c] != "unknown").sum()
        else:
            filled = tr[c].notna().sum()
        print(f"  {label:<26} {filled:>4} / {len(tr)}")
    print("-" * 70)
    print("STREAM x success (training pool)")
    g = tr.groupby("Stream")["is_success"].agg(["size", "sum", "mean"])
    g["mean"] = (g["mean"] * 100).round(1)
    g.columns = ["n", "wins", "win_rate_%"]
    print(g.to_string())
    print("-" * 70)
    print("Top placements x success (training pool, n>=5)")
    gp = tr.groupby("placement_clean")["is_success"].agg(["size", "sum", "mean"])
    gp = gp[gp["size"] >= 5].sort_values("size", ascending=False)
    gp["mean"] = (gp["mean"] * 100).round(1)
    gp.columns = ["n", "wins", "win_rate_%"]
    print(gp.to_string())
    print("=" * 70)


if __name__ == "__main__":
    df = build()
    keep = [
        "ID", "Experiment Key", "Name", "Stream",
        "placement_clean", "kpi_family", "Stage",
        "Expected relative uplift", "expected_profit", "actual_profit",
        "duration_days", "start_year",
        "is_active", "is_pretest", "in_training",
        "isSuccessful", "is_success",
    ]
    out = df[keep].copy()
    out.to_csv(OUT, index=False)
    data_map(df)
    print(f"\nWrote {OUT.relative_to(HERE)}  ({len(out)} rows, {len(keep)} cols)")
