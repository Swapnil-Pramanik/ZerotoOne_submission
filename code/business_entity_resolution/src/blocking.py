"""Candidate generation: for every S2/S3 record, a short list of plausible S1 entities.

Per country, one sparse product scores every (record, S1) pair by the IDF-weighted
overlap of blocking keys (records.block_keys, generated per chunk by records.keys_for): name tokens, name-token pairs,
name x place, number x place and place pairs. Keys held by too many S1 records are
dropped; combinations stay rare even when each part is common. The top-k S1 entities
per record are kept (plus a relative-score cut).
Records left with no candidate fall back to character-trigram TF-IDF on the
spaceless name (domain-style names, heavy rewrites).
"""

import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from records import keys_for



@dataclass
class BlockingConfig:
    key_cap_frac: float = 2e-3     # drop keys held by more than this share of a country's S1 ...
    key_cap_min: int = 50          # ... but never below key_cap_min
    key_cap_max: int = 400         # ... nor above key_cap_max (bounds cost and memory at full scale)
    top_k: int = 10
    rel_cut: float = 0.3           # drop candidates scoring < rel_cut * the record's best score
    fallback_k: int = 3
    fallback_max_df: float = 0.002  # char trigrams in more than this share of S1 names are ignored ...
    fallback_max_df_abs: int = 500  # ... and in more than this many
    chunk_budget: int = 20_000_000  # max estimated partial products per chunk (memory guard)
    workers: int = 0               # 0 = all cores; 1 = serial

    def key_cap(self, n1: int) -> int:
        return int(min(self.key_cap_max, max(self.key_cap_min, self.key_cap_frac * n1)))


# Keys are hashed (no Python vocabulary: ~25M distinct US keys as a dict cost ~10 GB). With 2^30
# buckets, ~25M keys collide at ~2%; colliding keys merge, which at worst adds a few candidates.
_HASHER = HashingVectorizer(n_features=2 ** 30, tokenizer=str.split, token_pattern=None, lowercase=False,
                            alternate_sign=False, norm=None, binary=True, dtype=np.float32)


def _remap(m: sp.csr_matrix, kept_cols: np.ndarray, weights: np.ndarray | None) -> sp.csr_matrix:
    """Keep only columns in kept_cols (sorted hashed ids), renumbered 0..K-1, optionally weighted."""
    rows = np.repeat(np.arange(m.shape[0], dtype=np.int32), np.diff(m.indptr))
    pos = np.searchsorted(kept_cols, m.indices)
    ok = pos < len(kept_cols)
    ok[ok] = kept_cols[pos[ok]] == m.indices[ok]
    data = weights[pos[ok]] if weights is not None else np.ones(int(ok.sum()), np.float32)
    return sp.csr_matrix((data, (rows[ok], pos[ok])), shape=(m.shape[0], len(kept_cols)), dtype=np.float32)


def key_matrices(s1c: pd.DataFrame, rc: pd.DataFrame, st, cap: int, chunk: int = 500_000):
    """(keys x S1) IDF-weighted matrix and (records x keys) binary matrix over the keys held by
    at most `cap` S1 records. Keys are generated and hashed chunk by chunk, never stored."""
    h1 = sp.vstack([_HASHER.transform(keys_for(s1c.iloc[i:i + chunk], st))
                    for i in range(0, len(s1c), chunk)]).tocsr()
    cols, df = np.unique(h1.indices, return_counts=True)
    keep = df <= cap
    kept, n1 = cols[keep], len(s1c)
    w = (np.log((n1 + 1) / (df[keep] + 1)) + 1.0).astype(np.float32)
    s1_m = _remap(h1, kept, w).T.tocsr()
    del h1, cols, df
    r_m = sp.vstack([_remap(_HASHER.transform(keys_for(rc.iloc[i:i + chunk], st)), kept, None)
                     for i in range(0, len(rc), chunk)]).tocsr()
    return s1_m, r_m


def topk_rows(c: sp.csr_matrix, k: int, rel_cut: float):
    """Row-wise top-k of a sparse score matrix -> (row, col, score) arrays."""
    c = c.tocsr()
    c.sum_duplicates()
    counts = np.diff(c.indptr)
    rows = np.repeat(np.arange(c.shape[0]), counts)
    order = np.lexsort((-c.data, rows))
    rows, cols, data = rows[order], c.indices[order], c.data[order]
    pos = np.arange(len(rows)) - c.indptr[rows]
    best = np.zeros(c.shape[0], dtype=np.float32)
    first = pos == 0
    best[rows[first]] = data[first]
    keep = (pos < k) & (data >= rel_cut * best[rows])
    return rows[keep], cols[keep], data[keep]


# Worker state: the S1 matrix is sent once per worker (initializer); each task ships one row chunk.
_WORKER = {}


def _init_worker(s1_m, k, rel_cut):
    _WORKER.update(s1_m=s1_m, k=k, rel_cut=rel_cut)


def _topk_chunk(args):
    start, r_chunk = args
    a, b, d = topk_rows(r_chunk @ _WORKER["s1_m"], _WORKER["k"], _WORKER["rel_cut"])
    return a + start, b, d


def _cost_chunks(r_m, s1_m, budget: int) -> list[tuple[int, int]]:
    """Row ranges whose estimated partial products (sum of key posting lengths) stay within budget."""
    posting = np.diff(s1_m.indptr).astype(np.float64)       # s1_m is keys x S1
    cost = np.cumsum(r_m @ posting)
    bounds, start = [], 0
    while start < r_m.shape[0]:
        base = cost[start - 1] if start else 0.0
        stop = int(np.searchsorted(cost, base + budget, side="right"))
        stop = min(max(stop, start + 1), r_m.shape[0])
        bounds.append((start, stop))
        start = stop
    return bounds


def sparse_topk(r_m, s1_m, k: int, rel_cut: float, budget: int, workers: int = 0):
    """Top-k columns of r_m @ s1_m per row, in cost-bounded row chunks across worker processes.

    Workers are started with 'forkserver' (clean processes), never 'fork': forking after
    PyTorch/CUDA has started threads can crash the children.
    """
    bounds = _cost_chunks(r_m, s1_m, budget)
    workers = min(workers or os.cpu_count() or 1, 3, len(bounds))
    if workers > 1:
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        parts = [None] * len(bounds)
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(s1_m, k, rel_cut)) as ex:
            # at most 2 chunks per worker in flight: submitting everything at once would copy
            # the whole record matrix into the task queue
            todo = iter(enumerate(bounds))
            pending = {}

            def submit_next():
                nxt = next(todo, None)
                if nxt is not None:
                    i, (a, b) = nxt
                    pending[ex.submit(_topk_chunk, (a, r_m[a:b]))] = i

            for _ in range(2 * workers):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    parts[pending.pop(fut)] = fut.result()
                    submit_next()
    else:
        _init_worker(s1_m, k, rel_cut)
        parts = [_topk_chunk((a, r_m[a:b])) for a, b in bounds]
        _WORKER.clear()
    if not parts:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return tuple(np.concatenate(z) for z in zip(*parts))


def _fallback(s1_ns, r_ns, k: int, budget: int, workers: int = 0, max_df: int = 500):
    """Character-trigram TF-IDF top-k for records that no key reached."""
    empty = (np.array([], int), np.array([], int), np.array([], np.float32))
    if not r_ns:
        return empty
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=max_df, dtype=np.float32)
    try:
        x1 = vec.fit_transform(s1_ns)
    except ValueError:  # vocabulary empty after pruning (tiny inputs)
        return empty
    return sparse_topk(vec.transform(r_ns), x1.T.tocsr(), k, 0.0, budget, workers)


def generate_candidates(s1: pd.DataFrame, r: pd.DataFrame, stats: dict, cfg: BlockingConfig = BlockingConfig(),
                        log=print) -> pd.DataFrame:
    """Candidate pairs as positional int32 indices: r_idx, s1_idx, block_score, fallback (bool)."""
    parts = []
    s1_groups = s1.groupby("country").indices
    for country, r_idx in r.groupby("country").indices.items():
        if country not in s1_groups:
            continue
        t0 = time.time()
        s1_idx = s1_groups[country]
        n1 = len(s1_idx)
        s1c, rc = s1.iloc[s1_idx], r.iloc[r_idx]

        s1_m, r_m = key_matrices(s1c, rc, stats.get(country), cfg.key_cap(n1))

        rows, cols, scores = sparse_topk(r_m, s1_m, cfg.top_k, cfg.rel_cut, cfg.chunk_budget, cfg.workers)

        empty = np.setdiff1d(np.arange(len(r_idx)), rows)
        fb_df = max(2, int(min(cfg.fallback_max_df * n1, cfg.fallback_max_df_abs)))
        fb = _fallback(s1c["core_ns"].tolist(), rc["core_ns"].to_numpy()[empty].tolist(), cfg.fallback_k,
                       cfg.chunk_budget, cfg.workers, fb_df)
        parts.append(pd.DataFrame({
            "r_idx": np.concatenate([r_idx[rows], r_idx[empty[fb[0]]]]).astype(np.int32),
            "s1_idx": np.concatenate([s1_idx[cols], s1_idx[fb[1]]]).astype(np.int32),
            "block_score": np.concatenate([scores, np.zeros(len(fb[0]), np.float32)]).astype(np.float32),
            "fallback": np.concatenate([np.zeros(len(rows), bool), np.ones(len(fb[0]), bool)]),
        }))
        log(f"  blocking {country}: {len(r_idx):,} records x {n1:,} S1 -> {len(parts[-1]):,} candidates "
            f"({len(parts[-1]) / len(r_idx):.2f}/record, {len(empty):,} via fallback) in {time.time() - t0:.0f}s")
    return pd.concat(parts, ignore_index=True)


def candidate_context(cand: pd.DataFrame) -> pd.DataFrame:
    """S1-side context over the *full* candidate set (so it is identical for sampled training rows)."""
    cand = cand.copy()
    g = cand.groupby("s1_idx")["block_score"]
    cand["s1_indegree"] = g.transform("size").astype(np.float32)
    cand["s1_block_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
    return cand
