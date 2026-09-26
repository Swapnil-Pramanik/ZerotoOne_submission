"""Alias tables learned from aligned training pairs (no external data).

Three kinds of alias are learned from true (S1, S2/S3) pairs:

* token equivalences: when two matched names (or addresses) differ by exactly one
  token on each side, that token pair is a candidate synonym (pvt/private,
  ltd/limited, st/street, rd/road, ...). Frequent, consistent pairs that look like an
  abbreviation or spelling variant are merged with union-find, and each group maps to
  its most common spelling.
* transliterations: for non-Latin S2/S3 names with the same number of tokens as the
  S1 name, tokens are aligned by position (anyascii 'praivet' -> 'private').
* state aliases: an address end-component of the S2/S3 record that consistently
  co-occurs with a given S1 state ('mh' -> 'maharashtra', 'pennsylvania' -> 'pa').

French street abbreviations are the only hand-written entries (language
normalisation, applied to France only, since there are no French training pairs). French
addresses get *only* that map: aliases learned on US/India addresses (st -> street) would
corrupt French ones (St-Nazaire -> "street nazaire").
"""

from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz

FRENCH_STREET = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "bvd": "boulevard",
    "pl": "place", "ch": "chemin", "che": "chemin", "imp": "impasse", "all": "allee", "rte": "route",
    "fbg": "faubourg", "crs": "cours", "sq": "square", "qu": "quai", "ndeg": "no",
    "st": "saint", "ste": "sainte",
}


def _has_digit(tok: str) -> bool:
    return any(c.isdigit() for c in tok)


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def _plausible_variant(x: str, y: str) -> bool:
    """Abbreviation or spelling variant (st/street, pvt/private, technolgy/technology), not a different
    word that merely co-occurs (south/delhi): same first letter, and short or very similar."""
    return x[0] == y[0] and (min(len(x), len(y)) <= 4 or fuzz.ratio(x, y) >= 80)


def learn_equivalences(a_texts, b_texts, freq: Counter, min_count=50, min_share=0.3) -> dict:
    """Token -> canonical token, from pairs of texts that differ by exactly one token each."""
    pair_counts, tok_counts = Counter(), Counter()
    for a, b in zip(a_texts, b_texts):
        ta, tb = set(a.split()), set(b.split())
        da, db = ta - tb, tb - ta
        if len(da) == 1 and len(db) == 1:
            x, y = next(iter(da)), next(iter(db))
            if _has_digit(x) or _has_digit(y):
                continue
            pair_counts[(x, y) if x < y else (y, x)] += 1
            tok_counts[x] += 1
            tok_counts[y] += 1
    uf = _UnionFind()
    for (x, y), c in pair_counts.items():
        if c >= min_count and c / min(tok_counts[x], tok_counts[y]) >= min_share and _plausible_variant(x, y):
            uf.union(x, y)
    groups = {}
    for tok in list(uf.parent):
        groups.setdefault(uf.find(tok), []).append(tok)
    mapping = {}
    for members in groups.values():
        canon = max(members, key=lambda t: (freq.get(t, 0), t))
        mapping.update({t: canon for t in members if t != canon})
    return mapping


def learn_transliterations(a_texts, b_texts, min_count=5, min_share=0.6) -> dict:
    """Transliterated token -> S1 token, aligned by position (same token count only)."""
    counts, totals = Counter(), Counter()
    for a, b in zip(a_texts, b_texts):
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(ta, tb):
            totals[y] += 1
            if x != y:
                counts[(y, x)] += 1
    best = {}
    for (y, x), c in counts.items():
        if c >= min_count and c / totals[y] >= min_share and c > best.get(y, (None, 0))[1]:
            best[y] = (x, c)
    return {y: x for y, (x, _) in best.items()}


def learn_states(s1_states, r_ends, min_count=50, min_share=0.9) -> tuple[dict, set]:
    """Map S2/S3 address end-components onto S1 states; also return the set of known S1 states."""
    state_freq = Counter(s1_states)
    known = {s for s, c in state_freq.items() if c >= min_count and s}
    counts, totals = Counter(), Counter()
    for state, ends in zip(s1_states, r_ends):
        for e in ends:
            if e and e != state:
                totals[e] += 1
                counts[(e, state)] += 1
    mapping = {}
    for (e, state), c in counts.items():
        if state in known and c >= min_count and c / totals[e] >= min_share:
            mapping[e] = state
    return mapping, known


@dataclass
class Aliases:
    name: dict = field(default_factory=dict)          # token -> canonical
    translit: dict = field(default_factory=dict)      # transliterated token -> S1 token
    address: dict = field(default_factory=dict)       # address token -> canonical
    state: dict = field(default_factory=dict)         # address end-component -> S1 state
    known_states: set = field(default_factory=set)

    def name_map(self) -> dict:
        """Transliteration first, then canonical spelling."""
        m = {t: self.name.get(v, v) for t, v in self.translit.items()}
        for t, v in self.name.items():
            m.setdefault(t, v)
        return m

    def address_map(self, country: str) -> dict:
        # aliases learned on US/India pairs must not touch French addresses ('st' is 'saint' there, not 'street')
        return dict(FRENCH_STREET) if country == "France" else dict(self.address)

    def summary(self) -> str:
        return (f"name equivalences {len(self.name):,}, transliterations {len(self.translit):,}, "
                f"address equivalences {len(self.address):,}, state aliases {len(self.state):,} "
                f"(known states {len(self.known_states):,})")
