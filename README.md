# A/B Test Toolkit

Two lightweight tools that help a product / growth team decide **which A/B tests to run** and **how to run them** — grounded in ~4 years of the company's historical experiments, and honest about what the data can and cannot tell you.

Built as a single Streamlit app; switch between the two tools in the sidebar.

---

## The problem it solves

Most teams prioritize test ideas from gut feeling, and size them with a one-off calculator each time. This toolkit turns both into a repeatable, evidence-based step:

- **Which idea first?** Rank a backlog by *expected value* instead of opinion.
- **Is it worth running?** See the sample, the run time, and the money at stake *before* you build.

---

## What's inside

### 1 · Hypothesis Prioritizer
Enter a **hypothesis**, a **feature description**, a **placement**, and the **target KPI**. You get:

- a **Priority score (0–100)** — the strength of the bet,
- a **confidence flag** (High / Medium / Low) — how much history actually backs it,
- and the **reasoning** spelled out, so it's a transparent reality check, not a black box.

Priority is a direct, proportional score of **Impact × Effort-efficiency**, where
**Impact = P(success) × relative value**:

- **P(success)** — a calibrated *base-rate anchor*: how often tests with this placement / KPI / feature type historically won. Metadata predicts winners only weakly (out-of-fold AUC ≈ 0.61), so this is honestly a prior, not a forecast of your specific idea.
- **Relative value** — the placement's pre-launch profit potential vs the site-wide typical, shown as an index (1.0× = typical). No absolute figures.
- **Effort-efficiency** — faster tests rank a bit higher, but it's a mild, subordinate lever.

If the text is thin, gibberish, or keyword-stuffed, the tool says so and caps its own confidence instead of returning a confident-looking guess.

### 2 · Test Planner
Pick a **product**, set the **baseline conversion rate** and **monthly traffic**, and choose an uplift range. For each detectable uplift the table shows:

- the **required sample** (α = 0.05, power = 0.80),
- the **days to run** at your traffic,
- and the **estimated annual $ impact**.

---

## Honest by design

Whether an A/B test wins is only weakly predictable from metadata — it's driven mostly by idea quality and execution, which aren't in the data. So the toolkit is built to:

- flag when an idea sits in a historically weak or strong segment,
- give a rough *relative* expected value,
- and be explicit about uncertainty — rather than pretend to predict winners.
