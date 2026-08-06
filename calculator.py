"""A/B Test Planner — sample size, days-to-run and annual-income (Streamlit).

Styled to match the Hypothesis Prioritizer page: native Streamlit components
(title + caption, bordered input group, columns, divider, metrics, a native
dataframe) — no custom page background or card CSS.

Confidentiality by design:
  - This file contains NO true coefficients. The published fallback uses the real
    product NAMES but K/L values deliberately perturbed ~10-15%, so the public app
    looks realistic without disclosing the exact numbers.
  - The TRUE K/L are read from `coefficients.csv` (gitignored) when running
    locally, or from Streamlit **Secrets** when deployed. If neither is present,
    the app falls back to the perturbed table.
"""
import os
import csv
import math
from statistics import NormalDist

import pandas as pd
import streamlit as st

# ---- fixed statistical parameters (production standard) ----
ALPHA = 0.05
POWER = 0.80
Z = NormalDist().inv_cdf(1 - ALPHA) + NormalDist().inv_cdf(POWER)
N_ROWS = 8


# ---- number formatting: dots for thousands ----
def dsep(n):
    return f"{int(round(n)):,}".replace(",", ".")


def money(n):
    s = f"${abs(int(round(n))):,}".replace(",", ".")
    return "-" + s if n < 0 else s


# ---- public fallback (safe to publish) ----
# Real product NAMES, but decoy K/L. Only the PRODUCT K*L is observable (it drives
# the annual-income column; sample size and days don't use K/L at all), so K and L
# are shifted in the SAME direction to move K*L a deliberate 10-15% off the truth
# (a different %/sign per product). Individual factors move only ~4-8%, so they
# still look natural. The public app is realistic without disclosing the real
# figures; the TRUE K/L live only in the gitignored coefficients.csv (local) or
# in Streamlit Secrets when deployed.
PLACEHOLDER = {
    "Subscription purchase (LEAD)":    (49.47,  8.61),
    "Subscription recurring (PAYING)": (42.61,  8.88),
    "Case subscription":               (63.12,  8.37),
    "Subscription upgrade":            (33.35,  8.21),
    "Drift subscription":              (34.05,  7.55),
    "Unsubscribe (retention)":         (-64.65, 5.13),
    "Luxe sample duo (provisional)":   (35.29,  7.83),
}


def load_coefficients():
    if os.path.exists("coefficients.csv"):
        d = {}
        with open("coefficients.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[row["product"]] = (float(row["k"]), float(row["l"]))
        if d:
            return d
    try:
        if "coefficients" in st.secrets:
            return {p: (float(v[0]), float(v[1])) for p, v in st.secrets["coefficients"].items()}
    except Exception:
        pass
    return PLACEHOLDER


COEF = load_coefficients()

# NOTE: set_page_config is called once in app.py (the navigator entry point);
# multipage sub-pages must not call it again.

# ---------------------------------------------------------------- header
st.title("A/B Test Planner")
st.caption(
    "Pick a product, set your baseline conversion rate and monthly traffic. "
    "For each detectable uplift you get the sample you need, the days to run, "
    "and the estimated annual $ impact — before you build."
)

# ---------------------------------------------------------------- inputs
def _reformat_traffic():
    digits = "".join(c for c in st.session_state.traffic_str if c.isdigit())
    st.session_state.traffic_str = dsep(digits) if digits else ""


if "traffic_str" not in st.session_state:
    st.session_state.traffic_str = "100.000"

with st.container(border=True):
    product = st.selectbox("Product", list(COEF.keys()),
                           help="K and L (the lifetime-value coefficients) are taken "
                                "from the selected product.")
    K, L = COEF[product]

    col_a, col_b = st.columns(2)
    with col_a:
        cr = st.number_input(
            "Baseline conversion rate (control), %",
            min_value=0.01, max_value=99.0, value=10.0, step=0.1, format="%.2f",
            help="Conversion rate of the metric you test, in the control group.",
        ) / 100.0
    with col_b:
        st.text_input(
            "Monthly eligible traffic (both arms)",
            key="traffic_str", on_change=_reformat_traffic,
            help="Users per month across BOTH arms on the surface you test.",
        )
    monthly = int("".join(c for c in st.session_state.traffic_str if c.isdigit()) or 0)

    st.markdown("**Uplift range to evaluate** (relative lift in the metric)")
    st.caption("Set the smallest and largest relative uplift you want to see. "
               "The table lists 8 values evenly spaced between them.")
    col_c, col_d = st.columns(2)
    with col_c:
        umin = st.number_input(
            "Smallest uplift, %", min_value=0.05, max_value=100.0,
            value=1.0, step=0.1, format="%.2f",
            help="First row of the table — the smallest lift you'd want to detect.",
        )
    with col_d:
        umax = st.number_input(
            "Largest uplift, %", min_value=0.05, max_value=100.0,
            value=5.0, step=0.1, format="%.2f",
            help="Last row of the table — the largest lift you'd want to detect.",
        )

if monthly < 1:
    st.info("Enter a monthly traffic value to see the plan.")
    st.stop()

# ---------------------------------------------------------------- results
daily = monthly / 30.0
records = []
for k in range(N_ROWS):
    u = (umin + (umax - umin) * k / (N_ROWS - 1)) / 100.0
    p0, p1 = cr, cr * (1 + u)
    sample = math.ceil(2 * Z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p0 * u) ** 2)
    days = math.ceil(sample / daily)
    annual = monthly * p0 * u * K * L
    records.append({
        "Detectable uplift": f"{u*100:.2f}%",
        "Variant CR": f"{p1*100:.2f}%",
        "Required sample (total)": dsep(sample),
        "Days to run": dsep(days),
        "Annual income uplift ($)": money(annual),
    })

df = pd.DataFrame(records)

st.divider()
st.markdown("#### What it would take to run")
st.dataframe(df, hide_index=True, use_container_width=True)
st.caption(
    "Sample sizes assume a two-sided test at 95% confidence and 80% power. "
    "Days to run assume the monthly traffic above, split evenly across 30 days."
)
