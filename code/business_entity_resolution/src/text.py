"""Text normalisation shared by every stage.

Everything is lowercase ASCII after this module: non-Latin scripts and accents are
transliterated with anyascii (only for rows that need it), and punctuation runs
become single spaces.
"""

import re

import pandas as pd
from anyascii import anyascii

# Indic scripts (Devanagari .. Malayalam) and Arabic, as literal-character ranges that
# work in both Python's re and pyarrow's RE2
NON_LATIN = "[ऀ-෿؀-ۿ]"
DOMAIN = r"(?i)(?:\.(?:com|net|org|in|co|fr|biz|info)\b|^www\.)"
_DIGITS = re.compile(r"\d+")
_DOTTED = re.compile(r"\b([a-z])\.(?=[a-z]\b)")   # s.a.r.l. -> sarl., l.l.c -> llc, p.o. -> po.


def ascii_lower(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype("string[pyarrow]")
    nonascii = s.str.contains(r"[^\x00-\x7F]")
    if nonascii.any():
        s = s.copy()
        s[nonascii] = [anyascii(x) for x in s[nonascii]]
    return s.str.lower()


def normalize(s: pd.Series) -> pd.Series:
    """Transliterate, lowercase, join dotted initials (S.A.R.L. -> sarl), collapse every
    non-alphanumeric run to one space."""
    s = ascii_lower(s)
    dotted = s.str.contains(r"(?:^|[^a-z0-9])[a-z]\.[a-z](?:[^a-z0-9]|$)")
    if dotted.any():
        s = s.copy()
        s[dotted] = [_DOTTED.sub(r"\1", x) for x in s[dotted]]
    return s.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def comma_part(s: pd.Series, which: int) -> pd.Series:
    """Normalised first (0) or last (-1) comma-separated component of an address."""
    pat = r"^(?P<part>[^,]*)" if which == 0 else r"(?P<part>[^,]*)$"
    return normalize(s.fillna("").str.extract(pat)["part"])


def numbers(text: str) -> str:
    """Space-joined sorted set of digit runs with leading zeros removed ('0227' -> '227')."""
    return " ".join(sorted({str(int(d)) for d in _DIGITS.findall(text)}))


def alpha_tokens(text: str, min_len: int = 2) -> str:
    return " ".join(t for t in text.split() if len(t) >= min_len and not any(c.isdigit() for c in t))


def map_tokens(texts, mapping: dict) -> list[str]:
    """Replace every token that has an entry in `mapping` (dropping tokens mapped to '')."""
    if not mapping:
        return list(texts)
    get = mapping.get
    out = []
    for t in texts:
        toks = [get(w, w) for w in t.split()]
        out.append(" ".join(w for w in toks if w))
    return out


def initials(text: str) -> str:
    return "".join(w[0] for w in text.split())
