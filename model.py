"""
Phase 2 / Tier 1 - P(success) model + calibration.

Tier 1 feature set (all available at prediction time — the PM supplies KPI and,
optionally, expected uplift, so there is no train/serve skew):

    P(success | placement, feature_type, kpi_family, expected_uplift)

  * Regularized logistic regression. Calibrated, interpretable, robust with ~94
    positives.
  * expected_uplift enters as log1p(min(uplift, 0.5)) — capped so a 190%-uplift
    outlier or an inflated claim can't dominate; regularization damps it further.
  * feature_type is kept for transparency even though it carries ~0 signal.

Deploy artifact: the fitted logistic-regression is exported to
artifacts/model_coefficients.json (per-category log-odds + numeric
standardization), so the app scores in pure numpy — no sklearn, no pickle,
fully auditable.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tests_classified.csv"
ART = HERE / "artifacts"

CAT = ["placement_clean", "feature_type", "kpi_family"]
NUM = ["exp_uplift_log"]
# Cap on expected uplift = the 99th percentile of real historical predictions
# (~30%). Beyond this, values are data errors (the raw data had a 190% entry) or
# inflated claims, not genuine forecasts. Data-driven, not a round-number guess:
# it clips only ~0.8% of tests yet halves the gaming headroom vs a 50% cap.
UPLIFT_CAP = 0.30
EB_ALPHA = 20.0


def load_training() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["in_training"]].copy()
    df["placement_clean"] = df["placement_clean"].fillna("Unknown").astype(str)
    df["feature_type"] = df["feature_type"].fillna("unclassified").astype(str)
    df["kpi_family"] = df["kpi_family"].fillna("unknown").astype(str)
    up = pd.to_numeric(df["Expected relative uplift"], errors="coerce").clip(upper=UPLIFT_CAP)
    df["exp_uplift_log"] = np.log1p(up)
    df["y"] = df["is_success"].astype(int)
    return df


def make_model(C: float) -> Pipeline:
    prep = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("nz", Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler())]), NUM),
    ])
    return Pipeline([("prep", prep),
                     ("lr", LogisticRegression(C=C, max_iter=2000))])


def eb_rate(w, n, prior, alpha=EB_ALPHA):
    return (w + alpha * prior) / (n + alpha)


def calibration_table(y, p, bins=5):
    q = pd.qcut(p, bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "bin": q})
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                            actual=("y", "mean"))
    return g.round(3)


def export_coefficients(model: Pipeline) -> dict:
    prep = model.named_steps["prep"]
    ohe = prep.named_transformers_["oh"]
    numpipe = prep.named_transformers_["nz"]
    scaler = numpipe.named_steps["s"]
    imputer = numpipe.named_steps["i"]
    lr = model.named_steps["lr"]
    coefs = lr.coef_[0]
    intercept = float(lr.intercept_[0])

    out = {"intercept": intercept, "categorical": {}, "numeric": {},
           "uplift_cap": UPLIFT_CAP, "cat_features": CAT}
    idx = 0
    for col, cats in zip(CAT, ohe.categories_):
        out["categorical"][col] = {str(c): float(coefs[idx + j]) for j, c in enumerate(cats)}
        idx += len(cats)
    for j, col in enumerate(NUM):
        out["numeric"][col] = {"mean": float(scaler.mean_[j]),
                               "std": float(scaler.scale_[j]),
                               "coef": float(coefs[idx + j]),
                               "impute": float(imputer.statistics_[j])}
    return out


def main():
    df = load_training()
    X, y = df[CAT + NUM], df["y"].values
    g_rate = y.mean()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    best = None
    for C in [0.15, 0.2, 0.3, 0.5, 1.0]:
        p = cross_val_predict(make_model(C), X, y, cv=cv, method="predict_proba")[:, 1]
        b = brier_score_loss(y, p)
        if best is None or b < best[1]:
            best = (C, b)
    C = best[0]

    p_oof = cross_val_predict(make_model(C), X, y, cv=cv, method="predict_proba")[:, 1]
    brier = brier_score_loss(y, p_oof)
    auc = roc_auc_score(y, p_oof)
    ll = log_loss(y, p_oof)
    brier_base = brier_score_loss(y, np.full_like(p_oof, g_rate))

    print("=" * 64)
    print(f"Training pool: {len(df)} | base rate {g_rate:.3f} | chosen C {C}")
    print(f"Features: {CAT + NUM}")
    print("-" * 64)
    print("OUT-OF-FOLD (5-fold CV)")
    print(f"  Brier   {brier:.4f}  (baseline {brier_base:.4f}, "
          f"skill {(1-brier/brier_base)*100:.1f}%)")
    print(f"  ROC-AUC {auc:.3f}   Log-loss {ll:.4f}")
    print("-" * 64)
    print("CALIBRATION (out-of-fold)")
    print(calibration_table(y, p_oof).to_string())
    print("-" * 64)

    model = make_model(C).fit(X, y)
    coeffs = export_coefficients(model)
    ART.mkdir(exist_ok=True)
    (ART / "model_coefficients.json").write_text(json.dumps(coeffs, indent=2), encoding="utf-8")

    # empirical-Bayes base rates for fallback + display
    def eb_by(col):
        gg = df.groupby(col)["y"].agg(["size", "sum"])
        gg["eb"] = [eb_rate(w, n, g_rate) for n, w in zip(gg["size"], gg["sum"])]
        return gg
    eb_place = eb_by("placement_clean")
    eb_kpi = eb_by("kpi_family")
    base_rates = {
        "global": round(float(g_rate), 4),
        "by_placement": {k: round(float(v), 4) for k, v in eb_place["eb"].items()},
        "by_kpi": {k: round(float(v), 4) for k, v in eb_kpi["eb"].items()},
        "placement_counts": {k: int(v) for k, v in eb_place["size"].items()},
        "kpi_counts": {k: int(v) for k, v in eb_kpi["size"].items()},
    }
    (ART / "base_rates.json").write_text(json.dumps(base_rates, indent=2), encoding="utf-8")

    meta = {"chosen_C": C, "n_train": int(len(df)), "base_rate": round(float(g_rate), 4),
            "brier_oof": round(float(brier), 4), "brier_baseline": round(float(brier_base), 4),
            "auc_oof": round(float(auc), 3), "features": CAT + NUM, "uplift_cap": UPLIFT_CAP}
    (ART / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # show the strongest coefficients (interpretability)
    print("Top +/- log-odds by category (relative within feature):")
    for col in CAT:
        d = coeffs["categorical"][col]
        srt = sorted(d.items(), key=lambda kv: kv[1])
        print(f"  {col}: lowest {srt[0][0]}={srt[0][1]:+.2f} | "
              f"highest {srt[-1][0]}={srt[-1][1]:+.2f}")
    print(f"  uplift coef (per +1 sd of log-uplift): {coeffs['numeric']['exp_uplift_log']['coef']:+.2f}")
    print("=" * 64)
    print("Wrote model_coefficients.json, base_rates.json, model_meta.json")


if __name__ == "__main__":
    main()
