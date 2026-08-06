"""
A/B Test Toolkit — entry point / navigator.

Two tools live under one app (sidebar switches between them):
  - Hypothesis Prioritizer  (home.py)      — rank test ideas by expected value
  - Sample-Size Calculator  (calculator.py) — sample, days-to-run and $ impact

Streamlit multipage note: st.set_page_config runs ONCE here; the individual
pages must not call it. Deploy still points at app.py.

Run locally:   streamlit run app.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# make imports (classify, score) resolve regardless of the launch cwd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

st.set_page_config(page_title="A/B Test Toolkit", layout="centered")

prioritizer = st.Page(
    "home.py", title="Hypothesis Prioritizer", default=True,
)
calculator = st.Page(
    "calculator.py", title="Test Planner",
)

st.navigation([prioritizer, calculator]).run()
