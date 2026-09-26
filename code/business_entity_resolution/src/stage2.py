"""Stage 2: sibling evidence, and per-entity match-set selection.

Most missed matches are records that barely resemble the S1 name but closely resemble
the entity's *other* records (same address and house numbers, similar name). Stage 1
assigns records to entities; for every uncertain pair (s1, r), stage 2 compares r with the
records already assigned to s1 (its "siblings") and a second LightGBM re-scores the pair.

Set selection replaces one global threshold: for each entity it keeps the top-k of the
records assigned to it, choosing k (possibly 0) to maximise the expected per-entity F0.5.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

SIBLING_FEATURES = ["sib_n", "sib_n_same_source", "sib_name", "sib_addr", "sib_num", "sib_p1_max"]
SIB_PROB = 0.5        # a record counts as an entity's sibling when assigned to it with p1 >= SIB_PROB
SCOPE_MIN = 0.01      # stage 2 re-scores pairs with p1 >= SCOPE_MIN; others keep p1


def best_assignment(s1_idx: np.ndarray, r_idx: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that are their record's best-scoring candidate."""
    order = np.lexsort((-p, r_idx))
    first = np.ones(len(order), bool)
    first[1:] = r_idx[order][1:] != r_idx[order][:-1]
    mask = np.zeros(len(p), bool)
    mask[order[first]] = True
    return mask


def sibling_features(s1_idx, r_idx, p1, scope: np.ndarray, r: pd.DataFrame, chunk: int = 4_000_000) -> pd.DataFrame:
    """Sibling features for rows in `scope` (NaN elsewhere). Arrays are aligned candidate rows."""
    n = len(p1)
    out = {c: np.full(n, np.nan, np.float32) for c in SIBLING_FEATURES}
    sib = best_assignment(s1_idx, r_idx, p1) & (p1 >= SIB_PROB)
    siblings = pd.DataFrame({"s1_idx": s1_idx[sib], "sib_r": r_idx[sib], "sib_p": p1[sib]})
    rows = np.flatnonzero(scope)
    q = pd.DataFrame({"row": rows, "s1_idx": s1_idx[rows], "r_idx": r_idx[rows]})
    for c in SIBLING_FEATURES:
        out[c][rows] = 0.0
    name = r["name_a"]
    addr = r["addr_a"]
    nums = r["addr_nums"]
    src = r["source"].to_numpy()
    for start in range(0, len(q), chunk):
        qq = q.iloc[start:start + chunk]
        m = qq.merge(siblings, on="s1_idx")
        m = m[m["sib_r"].to_numpy() != m["r_idx"].to_numpy()]
        if m.empty:
            continue
        a, b = m["r_idx"].to_numpy(), m["sib_r"].to_numpy()
        f = pd.DataFrame({
            "row": m["row"].to_numpy(),
            "name": process.cpdist(name.take(a).tolist(), name.take(b).tolist(), scorer=fuzz.token_set_ratio,
                                   workers=-1, dtype=np.float32),
            "addr": process.cpdist(addr.take(a).tolist(), addr.take(b).tolist(), scorer=fuzz.token_set_ratio,
                                   workers=-1, dtype=np.float32),
            "num": np.fromiter((bool(set(x.split()) & set(y.split())) if x and y else False
                                for x, y in zip(nums.take(a).tolist(), nums.take(b).tolist())), bool, len(m)),
            "same": src[a] == src[b],
            "p": m["sib_p"].to_numpy(),
        })
        # empty addresses give no address evidence
        f.loc[(addr.take(a).str.len().to_numpy() == 0) | (addr.take(b).str.len().to_numpy() == 0), "addr"] = np.nan
        g = f.groupby("row")
        idx = g.size().index.to_numpy()
        out["sib_n"][idx] = g.size().to_numpy()
        out["sib_n_same_source"][idx] = g["same"].sum().to_numpy()
        out["sib_name"][idx] = g["name"].max().to_numpy()
        out["sib_addr"][idx] = g["addr"].max().to_numpy()
        out["sib_num"][idx] = g["num"].max().astype(np.float32).to_numpy()
        out["sib_p1_max"][idx] = g["p"].max().to_numpy()
    return pd.DataFrame(out)


def stage2_matrix(base: pd.DataFrame, p1: np.ndarray, sib: pd.DataFrame, r_idx: np.ndarray) -> pd.DataFrame:
    """Stage-1 features + p1 + sibling features + how this pair's sibling support compares with the
    record's other candidates (rank and margin), which is what resolves records that stage 1 gave
    to the wrong entity. `r_idx` must hold every scored candidate of each record."""
    X = base.reset_index(drop=True).copy()
    X["p1"] = p1.astype(np.float32)
    for c in SIBLING_FEATURES:
        X[c] = sib[c].to_numpy()
    support = (np.nan_to_num(X["sib_name"].to_numpy()) + np.nan_to_num(X["sib_addr"].to_numpy())
               + 100 * np.nan_to_num(X["sib_num"].to_numpy()))
    for name, v in (("sib_support", support), ("p1", X["p1"].to_numpy())):
        v = pd.Series(v)
        grp = v.groupby(r_idx)
        best = grp.transform("max").to_numpy()
        second = pd.Series(np.where(v.to_numpy() == best, -np.inf, v.to_numpy())).groupby(r_idx).transform("max")
        second = np.where(np.isfinite(second.to_numpy()), second.to_numpy(), 0.0)
        X[f"rec_{name}_rank"] = grp.rank(ascending=False, method="min").to_numpy(np.float32)
        X[f"rec_{name}_margin"] = np.where(v.to_numpy() == best, v.to_numpy() - second, v.to_numpy() - best).astype(np.float32)
    X["sib_support"] = support.astype(np.float32)
    return X


def select_sets(s1_idx, r_idx, p, min_p: float = 0.05) -> np.ndarray:
    """Per entity, keep the k best records (by p) that maximise expected F0.5; returns a row mask.

    Each record first goes to its best entity. For an entity with record probabilities
    p_1 >= p_2 >= ..., keeping the top k gives E[F0.5] ~ 1.25 * sum(p_1..p_k) / (0.25 * sum(p) + k),
    and keeping none scores prod(1 - p_i) (the chance the entity has no match among them).
    """
    best = best_assignment(s1_idx, r_idx, p) & (p >= min_p)
    rows = np.flatnonzero(best)
    d = pd.DataFrame({"row": rows, "e": s1_idx[rows], "p": p[rows]}).sort_values(["e", "p"], ascending=[True, False])
    g = d.groupby("e", sort=False)["p"]
    k = g.cumcount().to_numpy() + 1
    cum = g.cumsum().to_numpy()
    total = g.transform("sum").to_numpy()
    f_k = 1.25 * cum / (0.25 * total + k)
    log_none = pd.Series(np.log1p(-np.clip(d["p"].to_numpy(), 0, 0.999999)))
    empty = np.exp(log_none.groupby(d["e"].to_numpy()).transform("sum").to_numpy())
    best_f = pd.Series(f_k).groupby(d["e"].to_numpy()).transform("max").to_numpy()
    k_star = pd.Series(np.where(f_k == best_f, k, np.inf)).groupby(d["e"].to_numpy()).transform("min").to_numpy()
    keep = (k <= k_star) & (best_f > empty)
    mask = np.zeros(len(p), bool)
    mask[d["row"].to_numpy()[keep]] = True
    return mask
