"""Pair features for (S1 entity, S2/S3 record) candidates.

Groups (no country indicator — the model must transfer to France):
  name     fuzzy scores on alias-mapped / core / spaceless names, IDF-weighted token
           overlap, acronym match, domain & script flags
  address  shared house numbers (overlap, containment), fuzzy address scores,
           state agreement via learned aliases (unknown -> NaN, e.g. France)
  context  blocking score, the pair's rank and margin inside the record's candidate
           list, how many S1 entities share this name, and (from the full candidate set,
           see blocking.candidate_context) the S1 entity's in-degree and this pair's rank there
"""

import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from records import idf

CHUNK = 1_000_000


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _set_features(a_nums, b_nums, a_core, b_core, a_init, b_init, a_ns, b_ns, s1_state, r_state,
                  countries, stats, known_states):
    n = len(a_nums)
    f = {k: np.full(n, np.nan, np.float32) for k in (
        "num_jacc", "num_s1_in_r", "num_r_in_s1", "num_first_eq", "num_n_s1", "num_n_r",
        "idf_recall", "idf_precision", "acronym", "state_match")}
    for i in range(n):
        na, nb = a_nums[i].split(), b_nums[i].split()
        f["num_n_s1"][i], f["num_n_r"][i] = len(na), len(nb)
        if na and nb:
            sa, sb = set(na), set(nb)
            inter = len(sa & sb)
            f["num_jacc"][i] = inter / len(sa | sb)
            f["num_s1_in_r"][i] = inter / len(sa)
            f["num_r_in_s1"][i] = inter / len(sb)
            f["num_first_eq"][i] = float(na[0] in sb)
        st = stats.get(countries[i])
        ta, tb = set(a_core[i].split()), set(b_core[i].split())
        if ta and tb and st is not None:
            w = {t: idf(st.name_df.get(t, 0), st.n_s1) for t in ta | tb}
            shared = sum(w[t] for t in ta & tb)
            f["idf_recall"][i] = shared / sum(w[t] for t in ta)
            f["idf_precision"][i] = shared / sum(w[t] for t in tb)
        f["acronym"][i] = float((len(a_init[i]) >= 2 and a_init[i] == b_ns[i]) or
                                (len(b_init[i]) >= 2 and b_init[i] == a_ns[i]))
        rs = r_state[i].split("|") if r_state[i] else []
        if s1_state[i] in known_states and any(s in known_states for s in rs):
            f["state_match"][i] = float(s1_state[i] in rs)
    return f


def pair_features(s1: pd.DataFrame, r: pd.DataFrame, cand: pd.DataFrame, stats: dict, known_states: set,
                  log=print) -> pd.DataFrame:
    t0 = time.time()
    text_s1 = ("name_a", "name_core", "name_ns", "core_ns", "initials", "addr_a", "addr_words", "addr_nums",
               "state_s1", "country")
    text_r = ("name_a", "name_core", "name_ns", "core_ns", "initials", "addr_a", "addr_words", "addr_nums",
              "state_r", "source")
    flags_r = {c: r[c].to_numpy(bool) for c in ("has_addr", "non_latin", "is_domain")}
    # how many S1 entities in the same country share this core name (name gives no identity when large)
    key = s1["country"].astype(str) + "|" + s1["name_core"].astype(str)
    collision = key.map(key.value_counts()).to_numpy(np.float32)

    out = []
    for start in range(0, len(cand), CHUNK):
        c = cand.iloc[start:start + CHUNK]
        i1, ir = c["s1_idx"].to_numpy(), c["r_idx"].to_numpy()
        # take only this chunk's strings out of the compact arrow columns
        A = {k: np.array(s1[k].take(i1).tolist(), dtype=object) for k in text_s1}
        B = {k: np.array(r[k].take(ir).tolist(), dtype=object) for k in text_r}
        B.update({k: v[ir] for k, v in flags_r.items()})
        f = {
            "name_ratio": _cp(A["name_a"].tolist(), B["name_a"].tolist(), fuzz.ratio),
            "name_tset": _cp(A["name_a"].tolist(), B["name_a"].tolist(), fuzz.token_set_ratio),
            "name_tsort": _cp(A["name_a"].tolist(), B["name_a"].tolist(), fuzz.token_sort_ratio),
            "name_partial": _cp(A["name_a"].tolist(), B["name_a"].tolist(), fuzz.partial_ratio),
            "name_jw": _cp(A["name_a"].tolist(), B["name_a"].tolist(), JaroWinkler.normalized_similarity),
            "core_tset": _cp(A["name_core"].tolist(), B["name_core"].tolist(), fuzz.token_set_ratio),
            "ns_ratio": _cp(A["name_ns"].tolist(), B["name_ns"].tolist(), fuzz.ratio),
            "core_ns_partial": _cp(A["core_ns"].tolist(), B["core_ns"].tolist(), fuzz.partial_ratio),
            "addr_tset": _cp(A["addr_a"].tolist(), B["addr_a"].tolist(), fuzz.token_set_ratio),
            "addr_ratio": _cp(A["addr_a"].tolist(), B["addr_a"].tolist(), fuzz.ratio),
            "addr_words_tset": _cp(A["addr_words"].tolist(), B["addr_words"].tolist(), fuzz.token_set_ratio),
        }
        no_addr = ~B["has_addr"].astype(bool)
        for k in ("addr_tset", "addr_ratio", "addr_words_tset"):
            f[k][no_addr] = np.nan
        f.update(_set_features(A["addr_nums"], B["addr_nums"], A["name_core"], B["name_core"], A["initials"],
                               B["initials"], A["name_ns"], B["name_ns"], A["state_s1"], B["state_r"],
                               A["country"], stats, known_states))
        f.update({
            "name_tokens_s1": np.array([len(x.split()) for x in A["name_a"]], np.float32),
            "name_tokens_r": np.array([len(x.split()) for x in B["name_a"]], np.float32),
            "r_has_addr": B["has_addr"].astype(np.float32),
            "r_non_latin": B["non_latin"].astype(np.float32),
            "r_is_domain": B["is_domain"].astype(np.float32),
            "r_is_s3": (B["source"] == "S3").astype(np.float32),
            "s1_collision": collision[i1],
            "block_score": c["block_score"].to_numpy(np.float32),
            "fallback": c["fallback"].to_numpy(np.float32),
            "ctx_s1_indegree": c["s1_indegree"].to_numpy(np.float32),
            "ctx_s1_block_rank": c["s1_block_rank"].to_numpy(np.float32),
        })
        # GPU-model features (v2), present when the neural stages ran
        for col in ("dense_cos", "dense_rank", "dense_only", "ce_score"):
            if col in c:
                f[col] = c[col].to_numpy(np.float32)
        out.append(pd.DataFrame(f))
        log(f"  features: {min(start + CHUNK, len(cand)):,}/{len(cand):,} pairs ({time.time() - t0:.0f}s)")
    X = pd.concat(out, ignore_index=True)
    return add_context(X, cand)


def add_context(X: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Where does this S1 entity stand among the record's other candidates (and vice versa)?"""
    combo = X["name_tset"].to_numpy() + np.nan_to_num(X["addr_tset"].to_numpy(), nan=0.0) \
        + 100 * np.nan_to_num(X["num_s1_in_r"].to_numpy(), nan=0.0)
    g = pd.DataFrame({"r": cand["r_idx"].to_numpy(), "s1": cand["s1_idx"].to_numpy(), "combo": combo,
                      "block": X["block_score"].to_numpy()})
    grp = g.groupby("r")["combo"]
    top1 = grp.transform("max").to_numpy()
    # second best: max after masking one occurrence of the best
    is_top = combo == top1
    first_top = is_top & ~pd.Series(is_top).groupby(g["r"]).cumsum().gt(1).to_numpy()
    masked = np.where(first_top, -np.inf, combo)
    top2 = pd.Series(masked).groupby(g["r"]).transform("max").to_numpy()
    X["ctx_n_cands"] = grp.transform("size").to_numpy(np.float32)
    X["ctx_combo"] = combo.astype(np.float32)
    X["ctx_combo_rank"] = grp.rank(ascending=False, method="min").to_numpy(np.float32)
    X["ctx_margin"] = np.where(first_top, combo - np.where(np.isfinite(top2), top2, 0.0), combo - top1).astype(np.float32)
    X["ctx_block_rank"] = g.groupby("r")["block"].rank(ascending=False, method="min").to_numpy(np.float32)
    for col in ("ce_score", "dense_cos"):
        if col in X:  # the model score's standing within the record's candidate list
            v = pd.Series(np.nan_to_num(X[col].to_numpy(), nan=-1.0))
            best = v.groupby(g["r"]).transform("max").to_numpy()
            X[f"ctx_{col}_rank"] = v.groupby(g["r"]).rank(ascending=False, method="min").to_numpy(np.float32)
            X[f"ctx_{col}_gap"] = (v.to_numpy() - best).astype(np.float32)
    return X
