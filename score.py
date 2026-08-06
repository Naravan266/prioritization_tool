"""
Phase 3-4 / Tier 1 - the scoring engine.

Combines the calibrated P(success) and a RELATIVE value index into the composite
priority, with an honest confidence flag. Scores in pure numpy from the exported
logistic-regression coefficients — no sklearn at serve time.

    index_ev       = P(success) x value_ratio      (unitless; value_ratio = x typical)
    priority_0_100 = percentile of index_ev across historical segments
    ev_dollars     = P(success) x (PM's own profit input)   -- only when supplied

PRIVACY: no absolute company $ is stored or shown. Value is a relative index
(placement median / global median); an absolute expected value in $ appears only
when the PM enters their own profit estimate (their data). Sparse / OOD input ->
unknown categories contribute 0 log-odds (fallback toward the base rate) + LOW
confidence.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

_COEF = json.loads((ART / "model_coefficients.json").read_text(encoding="utf-8"))
_BASE = json.loads((ART / "base_rates.json").read_text(encoding="utf-8"))
_VALUE = json.loads((ART / "value_table.json").read_text(encoding="utf-8"))
_META = json.loads((ART / "model_meta.json").read_text(encoding="utf-8"))

GLOBAL_RATE = _BASE["global"]
MIN_VALUE_SUPPORT = _VALUE["min_support"]
UPLIFT_CAP = _COEF["uplift_cap"]

# Effort adjustment: a fast/cheap test is worth more per unit of work. Priority
# ranks on Impact x Efficiency, where Efficiency = (BASELINE_DAYS / dev_days)^w.
# If the PM doesn't enter their own days, we estimate them from the feature type.
#
# flow corrected 9 -> 4 from the real Dev Estimates in the test briefs (flow tests
# actually took 2-4 days, not 9). messaging/visual/personalization have no brief
# data yet, so they remain estimates.
FEATURE_DEV_DAYS = {
    "messaging": 2.0,        # copy / text changes - cheap (domain estimate, no data)
    "visual": 2.0,           # layout / widget tweaks (domain estimate, no data)
    "flow": 4.0,             # steps / checkout - from real brief estimates (was 9)
    "unclassified": 3.5,     # unknown type -> assume the typical build time (neutral)
    "personalization": 8.0,  # recommendations / dynamic logic (domain estimate, no data)
}
# EFFORT_WEIGHT (w) = how strongly dev speed influences priority. w=0 ignores days,
# w=1 is a full RICE-style division (days dominate). We use 0.3 so days stay a
# real but SUBORDINATE lever to placement/KPI - justified because the day figures
# are the least reliable input (rough per-type estimates; flow was off ~2-3x).
EFFORT_WEIGHT = 0.3
# Neutral point (1.0x) = the TYPICAL test build time in the data: the mean
# per-type estimate across all 484 tests is 3.43 days, and the median of the real
# Dev Estimates in the test briefs is 3.5 days - two methods converging on ~3.5.
# (Note it also CANCELS out of the ranking, so it only sets where "1.0x" reads.)
BASELINE_DAYS = 3.5
EFFICIENCY_CLAMP = (0.5, 2.0)


def _efficiency(days: float) -> float:
    e = (BASELINE_DAYS / max(float(days), 1.0)) ** EFFORT_WEIGHT
    return max(EFFICIENCY_CLAMP[0], min(EFFICIENCY_CLAMP[1], e))

CANON_PLACEMENTS = sorted(_COEF["categorical"]["placement_clean"].keys())
FEATURE_TYPES = sorted(_COEF["categorical"]["feature_type"].keys())
KPI_FAMILIES = sorted(_COEF["categorical"]["kpi_family"].keys())

# curated PM dropdowns
PLACEMENT_CHOICES = sorted(
    [p for p, n in _BASE["placement_counts"].items()
     if ("," not in p) and ("&" not in p) and p != "Unknown" and n >= 5],
    key=lambda p: -_BASE["placement_counts"][p],
)
OTHER_PLACEMENT = "Other / not listed"

# friendly labels for KPI families (only those the model knows)
KPI_LABELS = {
    "subscription": "Subscription purchase",
    "recurring": "Recurring charge",
    "upgrade_addon": "Upgrade / Add-on",
    "add_to_queue": "Add to queue",
    "case_sub": "Case subscription",
    "ecommerce": "Ecommerce purchase",
    "unsubscribe": "Unsubscribe (churn)",
    "opt_in": "Luxe Opt-in",
    "account": "Account action",
    "other": "Other",
    "unknown": "Not sure / other",
}
KPI_CHOICES = [(KPI_LABELS.get(k, k), k) for k in
               ["subscription", "recurring", "upgrade_addon", "add_to_queue",
                "case_sub", "ecommerce", "unsubscribe", "opt_in", "unknown"]
               if k in KPI_FAMILIES]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def predict_p(placement: str, feature_type: str, kpi_family: str,
              exp_uplift: float | None) -> float:
    """P(success) from exported logistic coefficients (unknown categories -> 0)."""
    z = _COEF["intercept"]
    z += _COEF["categorical"]["placement_clean"].get(str(placement), 0.0)
    z += _COEF["categorical"]["feature_type"].get(str(feature_type), 0.0)
    z += _COEF["categorical"]["kpi_family"].get(str(kpi_family), 0.0)
    nz = _COEF["numeric"]["exp_uplift_log"]
    x = math.log1p(min(exp_uplift, UPLIFT_CAP)) if exp_uplift not in (None, "") else nz["impute"]
    z += nz["coef"] * ((x - nz["mean"]) / nz["std"])
    return _sigmoid(z)


def _value_ratio(placement: str) -> tuple[float, int, str]:
    """Relative value index (x typical); no absolute $. Global = 1.0."""
    pv = _VALUE["by_placement"].get(placement)
    if pv and pv["n"] >= MIN_VALUE_SUPPORT:
        return float(pv["ratio"]), int(pv["n"]), "placement_index"
    return 1.0, 0, "global_index"


def _confidence(placement_known, p_support, value_support, value_source,
                pm_input, class_level, kpi_known):
    reasons, pts = [], 0
    if p_support >= 25:
        pts += 2; reasons.append(f"{p_support} past tests on this placement")
    elif p_support >= 8:
        pts += 1; reasons.append(f"{p_support} past tests on this placement")
    else:
        reasons.append("few past tests on this placement")
    if kpi_known:
        pts += 1; reasons.append("target KPI specified")
    else:
        reasons.append("target KPI not specified")
    if value_source == "placement_index":
        pts += 1
        reasons.append(f"profit value based on {value_support} past tests here")
    else:
        reasons.append("profit value uses the site-wide typical (little data here)")
    if class_level == "high":
        pts += 1; reasons.append("feature type detected clearly")
    elif class_level == "none":
        reasons.append("feature type not detected from text")

    if not placement_known:
        return "Low", ["placement is outside the historical set"] + reasons
    level = "High" if pts >= 4 else "Medium" if pts >= 2 else "Low"
    return level, reasons


# ---- reference distribution for the 0-100 priority (on the relative index) ----
def _reference_index():
    # include the effort dimension so the 0-100 scale spans realistic
    # impact x efficiency values (cheap-fast winners up to expensive-slow ones)
    vals = []
    for pl in CANON_PLACEMENTS:
        r, _, _ = _value_ratio(pl)
        for kp in KPI_FAMILIES:
            for ft, days in FEATURE_DEV_DAYS.items():
                p = predict_p(pl, ft, kp, None)
                vals.append(p * r * _efficiency(days))
    return np.sort(np.array(vals))


_REF_SORTED = _reference_index()
# Direct (proportional) scale instead of a percentile rank: a percentile
# compresses the crowded middle, so a tiny impact difference (e.g. from the
# low-signal feature type) gets amplified into a big priority swing. A linear map
# of Impact -> 0..100 (outliers capped at the 95th pct) makes equal impact
# differences show as equal priority differences everywhere.
_REF_MIN = float(_REF_SORTED.min())
_REF_P95 = float(np.percentile(_REF_SORTED, 95))
_REF_T1 = float(np.percentile(_REF_SORTED, 33))   # tercile cutoffs for the tier
_REF_T2 = float(np.percentile(_REF_SORTED, 67))


def _priority_0_100(index_ev: float) -> int:
    frac = (index_ev - _REF_MIN) / (_REF_P95 - _REF_MIN)
    return int(round(float(np.clip(frac, 0.0, 1.0)) * 100))


def _tier(index_ev: float) -> str:
    """Low / Medium / High vs typical historical segments (rank-band lens)."""
    return "Low" if index_ev < _REF_T1 else ("Medium" if index_ev < _REF_T2 else "High")


def score(placement: str, feature_type: str, kpi_family: str = "unknown",
          exp_uplift: float | None = None, pm_profit: float | None = None,
          dev_days: float | None = None, dev_cost: float | None = None,
          class_level: str = "medium") -> dict:
    placement_known = placement in CANON_PLACEMENTS or placement in _BASE["placement_counts"]
    p = predict_p(placement, feature_type, kpi_family, exp_uplift)
    ratio, vsup, vsrc = _value_ratio(placement)
    impact = p * ratio                        # relative expected value (unitless)
    p_support = int(_BASE["placement_counts"].get(placement, 0))
    kpi_known = kpi_family not in (None, "", "unknown")
    pm_input = pm_profit not in (None, "") and pm_profit > 0
    ev_dollars = round(p * float(pm_profit), 0) if pm_input else None

    # ---- effort adjustment (value per unit effort) ----
    # PM's own days win; otherwise estimate from the feature type (grounded in the
    # team's typical build times). So effort ALWAYS factors in.
    pm_days = dev_days not in (None, "") and dev_days > 0
    if pm_days:
        eff_days, days_source = float(dev_days), "you"
    else:
        eff_days = FEATURE_DEV_DAYS.get(feature_type, BASELINE_DAYS)
        days_source = "feature-type estimate"
    eff = _efficiency(eff_days)
    adjusted = impact * eff
    value_per_day = round(impact / eff_days, 4)

    # ROI only when both $ profit and $ cost are supplied (their own numbers)
    has_cost = dev_cost not in (None, "") and dev_cost > 0
    roi = net_dollars = None
    if pm_input and has_cost:
        roi = round(ev_dollars / float(dev_cost), 2)
        net_dollars = round(ev_dollars - float(dev_cost), 0)

    conf, reasons = _confidence(placement_known, p_support, vsup, vsrc, pm_input,
                                class_level, kpi_known)
    return {
        "placement": placement, "feature_type": feature_type, "kpi_family": kpi_family,
        "p_success": round(p, 4), "p_success_pct": round(p * 100, 1),
        "value_ratio": round(ratio, 2), "value_source": vsrc,
        "impact_priority_0_100": _priority_0_100(impact),   # ignoring effort
        "priority_0_100": _priority_0_100(adjusted),         # effort-adjusted
        "tier": _tier(adjusted),                             # Low/Medium/High band
        "effort_used": True, "efficiency": round(eff, 2),
        "dev_days": round(eff_days, 1), "dev_days_source": days_source,
        "dev_cost": dev_cost if has_cost else None,
        "value_per_day": value_per_day,
        "ev_dollars": ev_dollars, "roi": roi, "net_dollars": net_dollars,
        "pm_input": pm_input,
        "confidence": conf, "confidence_reasons": reasons,
        "p_support": p_support, "value_support": vsup,
        "global_rate": GLOBAL_RATE,
        "uplift_used": exp_uplift, "kpi_known": kpi_known,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Placements: {len(PLACEMENT_CHOICES)} | KPIs: {[k for _,k in KPI_CHOICES]}")
    print(f"Global rate {GLOBAL_RATE} (value shown as relative index; $ only from PM input)\n")
    scenarios = [
        ("Payment", "messaging", "subscription", 0.02, None, "high"),
        ("TY", "flow", "upgrade_addon", 0.05, None, "high"),
        ("SP", "personalization", "subscription", None, None, "medium"),
        ("SP", "personalization", "upgrade_addon", 0.10, 400000, "medium"),
        ("ZZZ-new", "unclassified", "unknown", None, None, "none"),
    ]
    for pl, ft, kp, up, prof, cl in scenarios:
        r = score(pl, ft, kp, up, prof, class_level=cl)
        evd = f" EV=${r['ev_dollars']:,.0f}" if r["ev_dollars"] is not None else ""
        print(f"{pl}/{ft}/{kp} up={up} prof={prof}:")
        print(f"   P={r['p_success_pct']}%  value={r['value_ratio']}x  "
              f"priority={r['priority_0_100']}/100  conf={r['confidence']}{evd}")
    print("\n--- effort adjustment demo (Sitewide/upgrade_addon, impact fixed) ---")
    for days in [None, 3, 10, 30]:
        r = score("Sitewide", "unclassified", "upgrade_addon", None, None, dev_days=days)
        vpd = f" val/day={r['value_per_day']}" if r["value_per_day"] else ""
        print(f"   dev_days={str(days):<4} eff={r['efficiency']}x  "
              f"impact_prio={r['impact_priority_0_100']}  priority={r['priority_0_100']}/100{vpd}")
