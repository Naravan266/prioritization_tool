"""
A/B Test Hypothesis Prioritizer - PM-facing Streamlit tool.

A product manager enters a hypothesis, a feature description, and a placement,
and gets a priority score with the reasoning behind it. Everything the tool
knows comes from ~4 years of historical tests; it is an evidence-based reality
check, not an oracle.

Run locally:   streamlit run app.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# make imports work regardless of the launch cwd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from classify import classify, field_quality
from score import (score, PLACEMENT_CHOICES, OTHER_PLACEMENT, KPI_CHOICES)

# NOTE: set_page_config is called once in app.py (the navigator entry point);
# multipage sub-pages must not call it again.

# traffic-light colours for Low / Medium / High (confidence and tier)
LEVEL_COLORS = {"High": "#1a7f37", "Medium": "#c99700", "Low": "#c0392b"}


def level_badge(level: str) -> str:
    color = LEVEL_COLORS.get(level, "#666")
    return (f"<span style='background:{color};color:white;padding:3px 12px;"
            f"border-radius:12px;font-weight:600'>{level}</span>")


# ---------------------------------------------------------------- header
st.title("A/B Test Hypothesis Prioritizer")
st.caption(
    "Enter a hypothesis and where it lives. The tool scores it against ~4 years "
    "of past tests to help you rank your backlog — it estimates *expected value*, "
    "not a guarantee."
)

# ---------------------------------------------------------------- inputs
with st.form("inputs"):
    hypothesis = st.text_area(
        "Hypothesis",
        placeholder="e.g. Adding trust badges and a countdown timer near the "
                    "checkout button will increase subscriptions.",
        height=90,
    )
    feature = st.text_area(
        "Feature description",
        placeholder="e.g. New sticky footer with a discount offer and urgency copy.",
        height=70,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        placement = st.selectbox(
            "Placement (where the feature lives)",
            PLACEMENT_CHOICES + [OTHER_PLACEMENT],
        )
    with col_b:
        kpi_label = st.selectbox(
            "Target KPI (the metric you want to move)",
            [lbl for lbl, _ in KPI_CHOICES],
            help="You know your target metric — it adds real predictive signal.",
        )

    with st.expander("Optional: sharpen the estimate"):
        exp_uplift_pct = st.number_input(
            "Your expected relative uplift (%)", min_value=0.0, max_value=30.0,
            value=0.0, step=0.5,
            help="Leave 0 if unsure. Historically, tests expecting more uplift won "
                 "more often — but the value is capped at 30% (the 99th percentile "
                 "of real predictions) so an inflated claim can't dominate, and it "
                 "never raises the confidence flag.",
        )
        dev_days = st.number_input(
            "Dev effort (days to build)", min_value=0, value=0, step=1,
            help="Leave 0 to estimate it from the feature type. Faster tests "
                 "rank higher, but only mildly.",
        )

    submitted = st.form_submit_button("Score this hypothesis", type="primary")

# ---------------------------------------------------------------- scoring
if submitted:
    text = f"{hypothesis} {feature}".strip()
    if not text:
        st.warning("Enter at least a hypothesis or a feature description.")
        st.stop()

    cls = classify(text)

    # Per-field quality: check hypothesis and feature SEPARATELY so a good feature
    # can't mask a nonsense hypothesis in the concatenated text.
    FQ_MSG = {"gibberish": "doesn't look like real text",
              "too_short": "is too short to be meaningful"}
    field_issues = {"Hypothesis": field_quality(hypothesis),
                    "Feature description": field_quality(feature)}
    bad_fields = {f: q for f, q in field_issues.items() if q}

    # If NOTHING readable was entered (every non-empty field is junk), we cannot
    # score meaningfully — refuse rather than return the default-segment number.
    usable = [t for t in (hypothesis, feature) if t.strip() and not field_quality(t)]
    if not usable:
        st.error(
            "🚫 We couldn't read a meaningful hypothesis or feature description, so "
            "there's nothing to score. Describe the change in plain language — e.g. "
            "*“Add a countdown timer with urgency copy on the subscription page”* — "
            "and try again."
        )
        st.stop()

    for f, q in bad_fields.items():
        st.warning(f"⚠️ **{f}** {FQ_MSG[q]} — we couldn't validate it, so treat the "
                   f"result with extra caution.")

    kpi_family = next(k for lbl, k in KPI_CHOICES if lbl == kpi_label)
    res = score(
        placement, cls["primary"], kpi_family,
        exp_uplift=(exp_uplift_pct / 100.0) if exp_uplift_pct > 0 else None,
        dev_days=dev_days if dev_days > 0 else None,
        class_level=cls["confidence_level"],
    )

    # A nonsense hypothesis/feature can't be validated -> never show High confidence.
    if bad_fields:
        res["confidence"] = "Low"
        res["confidence_reasons"] = (
            [f"{f.lower()} {FQ_MSG[q]}" for f, q in bad_fields.items()]
            + res["confidence_reasons"]
        )
    # Keyword stuffing / ambiguous text -> we couldn't cleanly read the feature type,
    # so cap confidence at Medium (placement/KPI data is still solid, the text isn't).
    suspicious = {"possible_keyword_stuffing", "ambiguous_multi_category"} & set(cls.get("flags", []))
    if suspicious and res["confidence"] == "High":
        res["confidence"] = "Medium"
        res["confidence_reasons"] = ["feature type reading is unclear (keyword-dump / ambiguous)"] + res["confidence_reasons"]

    # ---- headline ----
    c1, c2 = st.columns([1, 1])
    with c1:
        prio_help = ("Direct 0–100 score of Impact × Efficiency (rewards big *and* "
                     "fast tests). Effort comes from your input, or is estimated "
                     "from the feature type when you leave it blank.")
        st.metric("Priority", f"{res['priority_0_100']} / 100", help=prio_help)
        st.caption("How strong the bet is (expected-value score).")
    with c2:
        st.markdown(
            f"**Confidence** &nbsp;<span title='How much history backs this "
            f"estimate — NOT the chance your test wins.'>ⓘ</span><br>"
            f"{level_badge(res['confidence'])}",
            unsafe_allow_html=True,
        )
        st.caption("How sure we are of this estimate — **not** the chance of winning.")

    st.divider()

    # ---- the composite, spelled out ----
    st.markdown("#### Why this score")
    cols = st.columns(2)
    cols[0].metric("P(success)", f"{res['p_success_pct']}%",
                   help="Calibrated historical win rate for this placement + KPI + "
                        "feature type (and your expected uplift, if given). A base-rate "
                        "anchor, not a prediction of your specific idea.")
    cols[1].metric("Relative value", f"{res['value_ratio']}× typical",
                   help="How the pre-launch profit potential of this placement compares "
                        "to the site-wide typical (1.0×). Shown as an index — absolute "
                        "figures are private.")

    src = "your input" if res["dev_days_source"] == "you" else "estimated from feature type"
    note = (f"<small>Impact = <b>P(success) × relative value</b> = "
            f"<b>{res['p_success_pct']}%</b> × <b>{res['value_ratio']}×</b>. "
            f"Effort factor <b>{res['efficiency']}×</b> "
            f"(<b>{res['dev_days']:.0f} dev-days</b>, {src}) — a mild, subordinate "
            f"lever vs placement/KPI.</small>")
    st.markdown(note, unsafe_allow_html=True)

    # ---- effort ----
    st.markdown("#### Effort")
    cols = st.columns(2)
    days_lbl = ("your estimate" if res["dev_days_source"] == "you"
                else f"est. from {res['feature_type']}")
    cols[0].metric("Dev effort", f"{res['dev_days']:.0f} days", help=days_lbl)
    cols[1].metric("Value / dev-day", f"{res['value_per_day']}",
                   help="Impact per day of build effort — compare across your backlog.")
    st.caption("⚠️ When you don't enter effort, it's estimated from the feature "
               "type. Your own number overrides it. Effort is a mild lever and "
               "never raises confidence.")

    st.divider()

    # ---- input-quality guardrails ----
    FLAG_MSG = {
        "possible_keyword_stuffing":
            "The text looks like a list of keywords rather than a sentence — "
            "the feature type was down-weighted.",
        "ambiguous_multi_category":
            "The text matches several feature types with no clear winner — "
            "confidence is capped.",
        "low_plausibility_text":
            "The text doesn't look like a real hypothesis — feature type ignored.",
        "very_short_input":
            "Too little text to read a feature type reliably.",
    }
    for f in cls.get("flags", []):
        if f in FLAG_MSG:
            st.warning("⚠️ " + FLAG_MSG[f])

    # ---- detected feature type (transparency) ----
    st.markdown("#### What the tool read from your text")
    if cls["primary"] == "unclassified":
        st.info("No feature-type keywords matched — treated as **Unclassified**. "
                "Add words describing the *mechanism* (e.g. 'simplify the flow', "
                "'trust copy', 'personalized recommendation') for a sharper read.")
    else:
        st.write(f"Detected feature type: **{cls['primary_name']}** "
                 f"(match confidence: {cls['confidence_level']})")
        if cls["secondary"]:
            st.caption(f"Also touches: {cls['secondary']['name']}")

    # ---- confidence reasoning ----
    st.markdown("#### How much to trust this")
    for r in res["confidence_reasons"]:
        st.markdown(f"- {r}")
    if res["confidence"] == "Low":
        st.warning("**Low confidence** — thin or out-of-distribution history. "
                   "Use this as a weak prior and lean on your own judgment.")

    # ---- honest caveat ----
    with st.expander("⚠️ Important: what this score can and cannot tell you"):
        st.markdown(
            f"""
Across 4 years of data, **whether an A/B test wins is only weakly predictable**
from placement, KPI and feature type. Success is driven mostly by idea quality
and execution, which aren't in the data.

So treat **P(success)** as a *base-rate anchor* ("tests like this historically
win about this often"), not a forecast of your specific idea. The ranking is
driven mainly by **relative value**. This tool is best at:
- flagging when an idea sits in a historically weak or strong segment,
- giving a rough relative expected value,
- and being honest about uncertainty — not at precisely predicting winners.
"""
        )

else:
    st.info("Fill in the fields above and click **Score this hypothesis**.")

st.divider()
st.caption(
    "Trained on ~4 years of past A/B tests · estimates are directional, "
    "not guarantees."
)
