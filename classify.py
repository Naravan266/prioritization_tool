"""
Phase 1 - Feature-type classifier (keyword lexicon).

classify(text) scores free text against the 4-category lexicon and returns the
primary category plus a *confidence* signal, so downstream code (and the PM UI)
can be honest when the evidence is thin or ambiguous.

Why keyword-first: the lexicon was given for exactly this task; it is transparent
("we matched these words -> this category"), reproducible, and easy to explain to
a non-technical PM. Confidence + no-match handling cover the weak spots.

Confidence levels:
  high   : >=2 distinct keywords for the winner AND margin>=2 over 2nd place
  medium : a clear winner (margin>=1) but thin evidence
  low    : single weak hit or a tie between categories
  none   : no keyword matched  -> 'unclassified' (an out-of-distribution signal)
"""

from __future__ import annotations
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEXICON_PATH = HERE / "artifacts" / "lexicon.json"

_LEXICON = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))

# Pre-compile one word-boundary regex per keyword. Multi-word phrases keep their
# internal spaces; \b on both ends avoids substring false hits (e.g. 'info' will
# not fire inside 'information').
def _compile(kw: str) -> re.Pattern:
    """Word-boundary match; allow an optional trailing 's' on single words so
    'badge'->'badges', 'plan'->'plans', 'testimonial'->'testimonials' all fire
    without maintaining explicit plurals. Phrases are matched verbatim."""
    esc = re.escape(kw)
    if " " not in kw and not kw.endswith("s"):
        esc += r"s?"
    return re.compile(r"\b" + esc + r"\b", re.IGNORECASE)


_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {}
for code, d in _LEXICON.items():
    _PATTERNS[code] = [(kw, _compile(kw)) for kw in d["keywords"]]

DISPLAY = {code: d["display_name"] for code, d in _LEXICON.items()}


_VOWELS = set("aeiou")


def _wordlike(tok: str) -> bool:
    """A token that plausibly reads as a real word (alphabetic, has a vowel)."""
    tok = tok.strip(".,;:!?()[]\"'").lower()
    return tok.isalpha() and len(tok) >= 2 and any(c in _VOWELS for c in tok)


# Small set of recognizable words for a per-field gibberish check. Detecting
# gibberish by vowels alone fails (keyboard mash like "qwerty"/"blah" has vowels),
# so we check how many tokens are actual known words: common English + every
# lexicon keyword + common product/UX terms.
_COMMON_WORDS = set("""
the a an and or to of in on for with by at from as is are be we our your their this
that it its will can not no yes new add remove show hide more less than when if then
so but into up down out over page user users test click tap screen home cart checkout
sign log login price offer discount free trial product products subscription upgrade
plan plans block section header footer nav menu search filter sort popup modal button
image text copy flow step steps simplify reduce personalize recommend price value cta
notification alert secure safe cancel anytime shipping description explain info urgency
timer countdown trust review reviews testimonial quiz payment sitewide main register
queue hero sticky rounded autofill express faster quicker direct combine merge skip
best tailored dynamic smart suggest match preference video gif animation icon color
layout design redesign banner badge feature change increase improve reassure near
""".split())
_KNOWN_WORDS = _COMMON_WORDS | {kw for pats in _PATTERNS.values() for kw, _ in pats
                                if " " not in kw}


def _recognizable(tok: str) -> bool:
    tok = tok.strip(".,;:!?()[]\"'").lower()
    return tok.isdigit() or tok in _KNOWN_WORDS


def field_quality(text: str | None) -> str | None:
    """Per-field sanity check (run on hypothesis and feature separately, so a
    normal feature can't mask a garbage hypothesis in the concatenated text).
    Returns 'too_short' | 'gibberish' | None.

    'gibberish' = 4+ tokens but almost none are recognizable words. This catches
    keyboard mash; it cannot catch coherent but off-topic prose (real words) -
    that needs semantics, and is a documented limitation."""
    toks = [t for t in (text or "").split() if t]
    n = len(toks)
    if n == 0:
        return None                       # emptiness handled by the caller
    if n < 2:                             # only a single word is too thin;
        return "too_short"                # 2-word features ("trust widget") are fine
    if (sum(_recognizable(t) for t in toks) / n) < 0.25:
        return "gibberish"
    return None


def _robustness_flags(text: str, scores: dict[str, int], margin: int) -> list[str]:
    """Detect abusive / low-quality input so the UI can refuse to sound confident.

    Catches: gibberish, near-empty input, keyword stuffing, and ambiguous
    multi-category dumps. It does NOT catch off-topic prose that happens to
    contain category words (e.g. a video-game description) - that needs semantic
    understanding, and is called out as a known limitation.
    """
    flags = []
    toks = [t for t in text.split() if t]
    n = len(toks)
    total = sum(scores.values())
    n_cats = sum(1 for v in scores.values() if v > 0)

    if n < 3:
        flags.append("very_short_input")
    if n >= 3 and (sum(_wordlike(t) for t in toks) / n) < 0.5:
        flags.append("low_plausibility_text")          # looks like gibberish
    # Stuffing = high keyword density AND spread across many categories. Dense
    # text in ONE category is just a concise, on-topic hypothesis, not stuffing.
    if total > 0 and n >= 4 and (total / n) > 0.5 and n_cats >= 3:
        flags.append("possible_keyword_stuffing")
    if n_cats >= 3 and margin <= 1:
        flags.append("ambiguous_multi_category")        # no clear winner
    return flags


def classify(text: str | None) -> dict:
    text = (text or "").lower()
    matched: dict[str, list[str]] = {}
    scores: dict[str, int] = {}
    for code, pats in _PATTERNS.items():
        hits = [kw for kw, pat in pats if pat.search(text)]
        matched[code] = hits
        scores[code] = len(hits)

    total = sum(scores.values())
    if total == 0:
        return {
            "primary": "unclassified",
            "primary_name": "Unclassified",
            "confidence_level": "none",
            "confidence_score": 0.0,
            "scores": scores,
            "matched_keywords": matched,
            "secondary": None,
            "flags": _robustness_flags(text, scores, 0),
        }

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_code, top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0
    margin = top - second
    share = top / total

    if top >= 2 and margin >= 2:
        level = "high"
    elif margin >= 1:
        level = "medium"
    else:
        level = "low"

    # numeric score: dominance (share) tempered by how much evidence there is
    conf = round(share * (1 - 0.5 ** top), 3)

    # Abusive / low-quality input must never read as confident. Stuffing or an
    # ambiguous multi-category dump caps confidence at 'low'; gibberish or a
    # near-empty field drops it to 'none' (the match is not trustworthy).
    flags = _robustness_flags(text, scores, margin)
    if {"low_plausibility_text", "very_short_input"} & set(flags):
        level, conf = "none", 0.0
    elif {"possible_keyword_stuffing", "ambiguous_multi_category"} & set(flags):
        level = "low"
        conf = min(conf, 0.2)

    secondary = None
    if len(ordered) > 1 and ordered[1][1] > 0:
        secondary = {"code": ordered[1][0], "name": DISPLAY[ordered[1][0]],
                     "score": ordered[1][1]}

    return {
        "primary": top_code,
        "primary_name": DISPLAY[top_code],
        "confidence_level": level,
        "confidence_score": conf,
        "scores": scores,
        "matched_keywords": {k: v for k, v in matched.items() if v},
        "secondary": secondary,
        "flags": flags,
    }


if __name__ == "__main__":
    import sys
    import pandas as pd
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    df = pd.read_csv(HERE / "data" / "tests_clean.csv")
    res = df["Name"].apply(classify)
    df["feature_type"] = res.apply(lambda r: r["primary"])
    df["feature_type_name"] = res.apply(lambda r: r["primary_name"])
    df["feature_conf_level"] = res.apply(lambda r: r["confidence_level"])
    df["feature_conf_score"] = res.apply(lambda r: r["confidence_score"])
    df["feature_matched"] = res.apply(
        lambda r: "; ".join(f"{k}:{'/'.join(v)}" for k, v in r["matched_keywords"].items())
    )
    df.to_csv(HERE / "data" / "tests_classified.csv", index=False)

    n = len(df)
    print("=" * 66)
    print(f"Classified {n} test names")
    print("-" * 66)
    print("FEATURE TYPE distribution (all rows)")
    print(df["feature_type"].value_counts().to_string())
    print("-" * 66)
    print("CONFIDENCE level distribution")
    print(df["feature_conf_level"].value_counts().to_string())
    unclass = (df["feature_type"] == "unclassified").mean() * 100
    print("-" * 66)
    print(f"NO-MATCH rate (unclassified): {unclass:.1f}%")
    print("-" * 66)
    print("Win rate by feature type (training pool)")
    tr = df[df["in_training"]]
    g = tr.groupby("feature_type")["is_success"].agg(["size", "sum", "mean"])
    g["mean"] = (g["mean"] * 100).round(1)
    g.columns = ["n", "wins", "win_rate_%"]
    print(g.to_string())
    print("-" * 66)
    print("Sample of unclassified names:")
    for nm in df.loc[df["feature_type"] == "unclassified", "Name"].head(15):
        print("   ", nm)
    print("=" * 66)
