"""Turn raw source rows into the normalised fields used by blocking and features.

`base_fields` is alias-free (it is also the input for learning aliases);
`finalize` applies aliases and per-country statistics of the S1 table
(filler words, token document frequencies) — so the same code works for
train, validation and test, and for a country never seen in training.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aliases import Aliases, learn_equivalences, learn_states, learn_transliterations
from text import DOMAIN, NON_LATIN, alpha_tokens, comma_part, initials, map_tokens, normalize, numbers

FILLER_SHARE = 0.005      # a name token in >= 0.5% of a country's S1 names is filler (llc, private, sarl, ...)
STOP_SHARE = 0.05         # ... in >= 5% is a stopword even for blocking-key combinations (limited, llc, sarl)
ADDR_COMMON_SHARE = 0.02  # an address word in >= 2% of a country's S1 addresses is too generic for blocking keys
KEY_NAME_TOKENS = 4       # name tokens used to build combination keys
KEY_ADDR_WORDS = 6        # address words used to build combination keys
KEY_NUMBERS = 3


def base_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Alias-free fields. Text is stored as compact arrow strings (Python str objects for
    ~12M records x 6 text columns would cost several GB)."""
    arrow = "string[pyarrow]"
    out = pd.DataFrame({
        "entity_id": pd.array(df["entity_id"].astype(str).to_numpy(), dtype=arrow),
        "country": pd.array(df["country"].astype(str).to_numpy(), dtype=arrow),
        "source": pd.array(df["source"].astype(str).to_numpy(), dtype=arrow) if "source" in df else "S1",
    })
    out["name_raw"] = df["name"].fillna("").astype(arrow).array
    out["addr_raw"] = df["address"].fillna("").astype(arrow).array
    out["name_n"] = normalize(df["name"]).astype(arrow).array
    out["addr_n"] = normalize(df["address"]).astype(arrow).array
    out["has_addr"] = df["address"].notna().to_numpy()
    out["non_latin"] = df["name"].fillna("").str.contains(NON_LATIN).to_numpy(dtype=bool)
    out["is_domain"] = df["name"].fillna("").str.contains(DOMAIN).to_numpy(dtype=bool)
    out["first_part"] = comma_part(df["address"], 0).astype(arrow).array
    out["last_part"] = comma_part(df["address"], -1).astype(arrow).array
    return out


def learn_aliases(s1_base: pd.DataFrame, r_base: pd.DataFrame, pairs: pd.DataFrame) -> Aliases:
    """Learn all alias tables from true pairs (pairs: s1, r entity ids). Only the needed
    columns of the paired rows are materialised, by position."""
    s1_ix = pd.Index(s1_base["entity_id"]).get_indexer(pairs["s1"])
    r_ix = pd.Index(r_base["entity_id"]).get_indexer(pairs["r"])

    def col(df, c, ix):
        return df[c].take(ix).tolist()

    name_freq = Counter(t for n in s1_base["name_n"] for t in n.split())
    addr_freq = Counter(t for n in s1_base["addr_n"] for t in n.split())

    al = Aliases()
    a_name, b_name = col(s1_base, "name_n", s1_ix), col(r_base, "name_n", r_ix)
    al.name = learn_equivalences(a_name, b_name, name_freq)
    nl = r_base["non_latin"].to_numpy()[r_ix]
    al.translit = learn_transliterations([x for x, k in zip(a_name, nl) if k], [x for x, k in zip(b_name, nl) if k])
    del a_name, b_name
    a_words = [alpha_tokens(t) for t in col(s1_base, "addr_n", s1_ix)]
    b_words = [alpha_tokens(t) for t in col(r_base, "addr_n", r_ix)]
    al.address = learn_equivalences(a_words, b_words, addr_freq)
    del a_words, b_words
    ends = list(zip(col(r_base, "first_part", r_ix), col(r_base, "last_part", r_ix)))
    al.state, al.known_states = learn_states(col(s1_base, "last_part", s1_ix), ends)
    return al


@dataclass
class CountryStats:
    n_s1: int
    filler: set
    stop: set
    name_df: dict        # core name token -> number of S1 records containing it
    addr_common: set


def country_stats(s1: pd.DataFrame) -> dict:
    """Per-country S1 statistics (computed on the scenario's own S1 table, incl. test/France)."""
    stats = {}
    for c, g in s1.groupby("country"):
        n = len(g)
        name_df = Counter(t for n_ in g["name_a"] for t in set(n_.split()))
        filler = {t for t, d in name_df.items() if d >= FILLER_SHARE * n}
        stop = {t for t, d in name_df.items() if d >= STOP_SHARE * n}
        addr_df = Counter(t for w in g["addr_words"] for t in set(w.split()))
        common = {t for t, d in addr_df.items() if d >= ADDR_COMMON_SHARE * n}
        stats[c] = CountryStats(n, filler, stop, {t: d for t, d in name_df.items() if t not in filler}, common)
    return stats


def block_keys(name: str, nums: str, words: str, stop: set, common: set) -> str:
    """Blocking keys: name tokens, name-token pairs, name x place, number x name, number x place, place pairs.

    Combinations stay rare even when each part is common ('shyam' + 'hubli'), which is
    what lets common names, number-less addresses and heavy rewrites be blocked.
    """
    nt = list(dict.fromkeys(t for t in name.split() if t not in stop))[:KEY_NAME_TOKENS]
    all_words = list(dict.fromkeys(w for w in words.split() if len(w) >= 3))
    ws = [w for w in all_words if w not in common][:KEY_ADDR_WORDS]
    places = (ws + [w for w in all_words if w in common])[:KEY_ADDR_WORDS]  # rare places first, then cities/states
    ns = nums.split()[:KEY_NUMBERS]
    keys = [f"n:{t}" for t in nt]
    keys += [f"nn:{a}|{b}" if a < b else f"nn:{b}|{a}" for i, a in enumerate(nt) for b in nt[i + 1:]]
    keys += [f"np:{t}|{w}" for t in nt[:3] for w in places]
    keys += [f"nm:{n}|{t}" for n in ns for t in nt[:3]]
    keys += [f"a:{n}|{w}" for n in ns for w in ws]
    keys += [f"aa:{a}|{b}" if a < b else f"aa:{b}|{a}" for i, a in enumerate(ws[:4]) for b in ws[i + 1:4]]
    return " ".join(keys)


def apply_aliases(base: pd.DataFrame, al: Aliases) -> pd.DataFrame:
    """Alias-mapped name/address text plus address numbers and words."""
    out = base.copy()
    out["name_a"] = map_tokens(out["name_n"], al.name_map())
    addr = np.empty(len(out), dtype=object)
    for c, idx in out.groupby("country").indices.items():
        addr[idx] = map_tokens(out["addr_n"].take(idx).tolist(), al.address_map(c))
    out["addr_a"] = addr
    out["addr_nums"] = [numbers(t) for t in out["addr_n"]]
    out["addr_words"] = [alpha_tokens(t) for t in out["addr_a"]]
    out["state_s1"] = out["last_part"]
    out["state_r"] = ["|".join(sorted({al.state.get(e, e) for e in (f, l) if e}))
                      for f, l in zip(out["first_part"], out["last_part"])]
    return out


def finalize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Add core-name, spaceless-name, initials and address blocking keys (needs country stats)."""
    core = np.empty(len(df), dtype=object)
    for c, idx in df.groupby("country").indices.items():
        st = stats.get(c)
        filler = st.filler if st else set()
        names = df["name_a"].to_numpy()[idx]
        core[idx] = [" ".join(t for t in n.split() if t not in filler) for n in names]
    df = df.copy()
    df["name_core"] = core
    df["name_ns"] = [n.replace(" ", "") for n in df["name_a"]]
    df["core_ns"] = [(c or n).replace(" ", "") for c, n in zip(df["name_core"], df["name_a"])]
    df["initials"] = [initials(c or n) for c, n in zip(df["name_core"], df["name_a"])]
    return df


def keys_for(df: pd.DataFrame, st) -> list[str]:
    """Blocking keys for a slice of one country's records (generated on demand, never stored:
    ~45 keys per record would cost several GB as a column)."""
    stop = st.stop if st else set()
    common = st.addr_common if st else set()
    return [block_keys(n, a, w, stop, common) for n, a, w in
            zip(df["name_a"].tolist(), df["addr_nums"].tolist(), df["addr_words"].tolist())]


def idf(df_count: int, n: int) -> float:
    return math.log((n + 1) / (df_count + 1)) + 1.0


# only what blocking, features and the neural models read, per table
KEEP_S1 = ["entity_id", "country", "name_raw", "addr_raw", "name_a", "name_core", "name_ns", "core_ns", "initials",
           "addr_a", "addr_words", "addr_nums", "state_s1"]
KEEP_R = ["entity_id", "country", "source", "has_addr", "non_latin", "is_domain", "name_raw", "addr_raw", "name_a",
          "name_core", "name_ns", "core_ns", "initials", "addr_a", "addr_words", "addr_nums", "state_r"]


def compact(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Keep only the listed columns, with text stored as compact arrow strings."""
    df = df[keep].copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.array(df[c].tolist(), dtype="string[pyarrow]")
    return df.reset_index(drop=True)
