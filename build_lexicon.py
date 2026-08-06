"""
Parse the feature-category keyword doc into a structured lexicon (lexicon.json).

Source: data/feature_categories.docx - one line per category:
    "1. <Category>: kw, kw, kw(s), phrase, ..."
We expand the "(x)" notation (e.g. review(s) -> review, reviews;
personalize(d) -> personalize, personalized) and keep multi-word phrases intact.
Keeping this as a build step means the lexicon is traceable to the source doc,
not hand-typed.
"""

from __future__ import annotations
import json
import re
import zipfile
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCX = HERE / "data" / "feature_categories.docx"
OUT = HERE / "artifacts" / "lexicon.json"

# short human-readable codes for the 4 categories, in doc order
CODES = ["messaging", "visual", "flow", "personalization"]

# Documented surface-form extensions: these are NOT new concepts, only common
# surface spellings of keywords already present in the doc (e.g. 'offer' /
# 'price clarity' -> discount/free trial/bogo; 'step(s)' -> one screen/less steps;
# 'layout'/'button' block -> block/plans). Kept separate so the writeup can show
# exactly what was added on top of the given lexicon and why.
EXTENSIONS = {
    "messaging": ["discount", "free trial", "bogo", "promo", "coupon", "deal",
                  "% off", "percent off", "opt-in", "opt in", "optin",
                  # domain vocabulary (from reading the 525 test names):
                  "social proof", "purchases of last", "selling fast", "best seller",
                  "best sellers", "member perks", "what you get", "faq",
                  "highlight", "highlighted", "processing within", "plan names",
                  "social media", "member perk", "guarantee"],
    "flow": ["one screen", "less steps", "fewer steps", "one step",
             "single step", "one page",
             # domain vocabulary:
             "burger menu", "menu", "nav", "apple pay", "applepay",
             "payment methods", "payment method", "register within",
             "create account", "cart auto-open", "auto-open", "quick payment",
             "confirmation pop-up", "confirmation popup", "short page"],
    "visual": ["block", "plans", "tariff", "plan block",
               # domain vocabulary (widgets / media / design):
               "pop-up", "popup", "widget", "storyly", "dash hudson", "cart widget",
               "carousel", "slider", "imitation", "loader", "logos"],
    "personalization": [
        # domain vocabulary (recommendations / dynamic / tailored):
        "constructor", "recommendations", "scent profile", "autosuggestion",
        "autosuggestions", "feed", "feeds", "quiz results", "for you"],
}


def read_docx_text(path: Path) -> str:
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8")
    xml = xml.replace("</w:p>", "\n")
    text = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(text)


def expand(term: str) -> list[str]:
    """review(s) -> [review, reviews]; personalize(d) -> [personalize, personalized]."""
    term = term.strip().lower()
    m = re.match(r"^(.*?)\(([a-z]+)\)(.*)$", term)
    if m:
        base, suf, tail = m.group(1), m.group(2), m.group(3)
        return [f"{base}{tail}".strip(), f"{base}{suf}{tail}".strip()]
    return [term]


def parse() -> dict:
    raw = read_docx_text(DOCX)
    # split the run-together text on "1.", "2." ... category markers.
    # NOTE: the numbers are glued to the previous word ("...call to action2."),
    # so we must NOT require whitespace before the digit. No feature keyword
    # contains a "<digit>." so this split is unambiguous.
    chunks = re.split(r"(\d)\.\s*", raw)
    # chunks -> ['', '1', 'Messaging...: kw,kw', '2', 'Visual...: kw', ...]
    cats = {}
    idx = 0
    for i in range(1, len(chunks) - 1, 2):
        num = chunks[i]
        body = chunks[i + 1].strip()
        if ":" not in body:
            continue
        name, kws = body.split(":", 1)
        terms = []
        for t in kws.split(","):
            terms.extend(expand(t))
        terms = sorted({t for t in terms if t})
        code = CODES[idx]
        ext = EXTENSIONS.get(code, [])
        cats[code] = {
            "display_name": name.strip(),
            "keywords": sorted(set(terms) | set(ext)),
            "core_keywords": terms,          # exactly from the doc
            "extension_keywords": sorted(ext),  # surface forms we added
        }
        idx += 1
    return cats


if __name__ == "__main__":
    cats = parse()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(cats, indent=2, ensure_ascii=False), encoding="utf-8")
    for code, d in cats.items():
        print(f"{code:<16} {d['display_name']:<32} {len(d['keywords'])} keywords")
        print("   ", ", ".join(d["keywords"][:12]), "...")
    print(f"\nWrote {OUT.relative_to(HERE)}")
