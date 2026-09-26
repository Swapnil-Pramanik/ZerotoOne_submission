"""v7: post-processing of a finished run (v6+): twin support + an LLM judge on the hardest pairs.

Input: the artifacts of a finished v6 run (cached validation and test scores, id tables,
decision settings, aliases). No model of the base run is retrained.

1. Rows to re-score: each record's best or second candidate with an uncertain probability.
2. Twin support (all countries, label-free): each such record's best look-alike among all
   other S2/S3 records, found with the pipeline's own blocking run record against record.
   Records of a real business come in "twins" (same address / house number, similar name);
   distractor records are loners, and distractors cause most false merges.
3. LLM judge (Qwen2.5-1.5B-Instruct) on the hardest subset only (France gets 65% of the budget): the business, up to
   three records already matched to it, and the candidate; P(Yes) vs P(No) as next token.
4. Combiners (logistic regression) fitted on the base run's validation cache: base only,
   + twins, + twins + LLM. The best by validation macro F0.5 (threshold re-tuned) is applied.
5. Output: the chosen submission, the base decision, the other combination and a
   France-only variant, and all scores for later use.
"""

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_pairs, load_split, write_id_lists
from evaluate import macro_f05
from llm_judge import Judge, JudgeConfig, prompt
from stage2 import best_assignment, select_sets


@dataclass
class V7Config:
    test_budget: int = 120_000       # hard test pairs sent to the LLM (~45 min at the probe's 44 pairs/s)
    val_budget: int = 30_000         # hard validation pairs for calibration (v6 base)
    france_share: float = 0.65       # share of the LLM test budget reserved for French pairs
    lo: float = 0.05                 # uncertainty band on the base probability ...
    hi: float = 0.95
    fr_lo: float = 0.02              # ... wider for France
    fr_hi: float = 0.985
    sibling_p: float = 0.9           # siblings shown to the LLM: records assigned to the entity with p >= this
    rescore_lo: float = 0.02         # rows the combiner may change: best / second candidate with p in [lo, hi]
    rescore_hi: float = 0.98
    use_llm: bool = True
    seed: int = 7


def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------- base run loading
def load_base(run_dir: Path) -> dict:
    """Normalise a v5 or v6 artifact folder into positions, probabilities and decision settings."""
    art = next(p.parent for p in run_dir.rglob("test_scores.parquet"))
    with open(art / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    sc = pd.read_parquet(art / "test_scores.parquet")
    if "s1_idx" in sc:  # v6
        s1_ids = pd.read_parquet(art / "test_s1_ids.parquet")
        r_ids = pd.read_parquet(art / "test_r_ids.parquet")["entity_id"].to_numpy()
        country = s1_ids["country"].to_numpy()
        s1_ids = s1_ids["entity_id"].to_numpy()
        s1_idx, r_idx = sc["s1_idx"].to_numpy(np.int32), sc["r_idx"].to_numpy(np.int32)
        method = bundle["method"]
        p = (sc["p2"] if method in ("stage2", "select") else sc["p1"]).to_numpy(np.float32)
        t = bundle["t2"] if method == "stage2" else bundle["t1"]
        val = pd.read_parquet(art / "val_scores.parquet") if (art / "val_scores.parquet").exists() else None
        if val is not None:
            val["p"] = val["p2"] if method in ("stage2", "select") else val["p1"]
    else:  # v5: ids per row
        s1_codes, s1_ids = pd.factorize(sc["s1"].astype(str))
        r_codes, r_ids = pd.factorize(sc["r"].astype(str))
        s1_idx, r_idx = s1_codes.astype(np.int32), r_codes.astype(np.int32)
        country = sc.groupby(s1_codes)["country"].first().astype(str).to_numpy()
        s1_ids, r_ids = np.asarray(s1_ids), np.asarray(r_ids)
        p = sc["prob"].to_numpy(np.float32)
        method, t, val = "stage1", bundle["threshold"], None
    return {"dir": art, "s1_ids": np.asarray(s1_ids, dtype=object), "r_ids": np.asarray(r_ids, dtype=object),
            "country": country, "s1_idx": s1_idx, "r_idx": r_idx, "p": p, "method": method, "t": t, "val": val,
            "bundle": bundle}


def decide(method, s1_idx, r_idx, p, t) -> np.ndarray:
    if method == "select":
        return select_sets(s1_idx, r_idx, p)
    return best_assignment(s1_idx, r_idx, p) & (p >= t)


# ---------------------------------------------------------------------------- hard pairs
def hard_pairs(s1_idx, r_idx, p, french: np.ndarray, cfg: V7Config, budget: int, t: float) -> np.ndarray:
    """Row indices of the hardest pairs: uncertain best candidates and close second candidates,
    closest to the decision threshold first, with a reserved French share."""
    order = np.lexsort((-p, r_idx))
    rank = np.empty(len(p), np.int32)
    first = np.ones(len(order), bool)
    first[1:] = r_idx[order][1:] != r_idx[order][:-1]
    starts = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank[order] = np.arange(len(order)) - starts
    lo = np.where(french, cfg.fr_lo, cfg.lo)
    hi = np.where(french, cfg.fr_hi, cfg.hi)
    cand = (rank <= 1) & (p >= lo) & (p <= hi)
    idx = np.flatnonzero(cand)
    urgency = np.abs(p[idx] - t)
    fr_idx = idx[french[idx]][np.argsort(urgency[french[idx]], kind="stable")]
    ot_idx = idx[~french[idx]][np.argsort(urgency[~french[idx]], kind="stable")]
    n_fr = min(len(fr_idx), int(budget * cfg.france_share))
    chosen = np.concatenate([fr_idx[:n_fr], ot_idx[:budget - n_fr]])
    if len(chosen) < budget:  # unused French or other budget goes to the rest
        rest = np.setdiff1d(np.concatenate([fr_idx, ot_idx]), chosen)
        chosen = np.concatenate([chosen, rest[np.argsort(np.abs(p[rest] - t), kind="stable")][:budget - len(chosen)]])
    return np.sort(chosen)


def sibling_lists(s1_idx, r_idx, p, rows, min_p: float, k: int = 3) -> list[np.ndarray]:
    """For each row in `rows`: up to k records assigned to its entity with p >= min_p (excluding itself)."""
    best = best_assignment(s1_idx, r_idx, p) & (p >= min_p)
    needed = np.isin(s1_idx, np.unique(s1_idx[rows]))
    keep = best & needed
    sib = pd.DataFrame({"e": s1_idx[keep], "r": r_idx[keep], "p": p[keep]}).sort_values(["e", "p"], ascending=[True, False])
    sib = sib[sib.groupby("e").cumcount().to_numpy() <= k]   # k + 1: one may be the candidate itself
    lists = {e: sib["r"].to_numpy()[ix] for e, ix in sib.groupby("e").indices.items()}
    empty = np.array([], np.int32)
    out = []
    for e, rr in zip(s1_idx[rows], r_idx[rows]):
        cand = lists.get(e, empty)
        out.append(cand[cand != rr][:k])
    return out


def texts(split: str):
    s1, r = load_split(split)
    f = lambda df: pd.Series((df["name"].fillna("") + " | " + df["address"].fillna("")).astype(str).to_numpy(),
                             index=df["entity_id"].astype(str).to_numpy())
    return f(s1), f(r)


def build_prompts(s1_ids, r_ids, s1_idx, r_idx, rows, sibs, S1T, RT, jcfg) -> list[str]:
    return [prompt(S1T[s1_ids[s1_idx[i]]], [RT[r_ids[x]] for x in sl], RT[r_ids[r_idx[i]]], jcfg)
            for i, sl in zip(rows, sibs)]


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


# ---------------------------------------------------------------------------- calibration
def val_ids_of_run(bundle) -> set:
    """Validation entity ids of a v6 run, by replaying its seeded fold assignment."""
    cfg = bundle["cfg"]
    rng = np.random.default_rng(cfg.seed)
    s1_raw, _ = load_split("train")
    ids = s1_raw["entity_id"].astype(str).to_numpy()
    ghost = rng.random(len(ids)) < cfg.ghost_frac
    kept = ids[~ghost]
    u = rng.random(len(kept))
    return set(kept[u < cfg.val_frac])


def rescore_rows(s1_idx, r_idx, p, lo: float, hi: float) -> np.ndarray:
    """Rows that post-processing may change: a record's best or second candidate with p in [lo, hi]."""
    order = np.lexsort((-p, r_idx))
    first = np.ones(len(order), bool)
    first[1:] = r_idx[order][1:] != r_idx[order][:-1]
    starts = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank = np.empty(len(p), np.int32)
    rank[order] = np.arange(len(order)) - starts
    return np.flatnonzero((rank <= 1) & (p >= lo) & (p <= hi))


def twin_features(split: str, rec_ids, aliases) -> pd.DataFrame:
    """Each record's best look-alike among all other S2/S3 records of the split ("twin").

    Records of a real entity come in twins (same address / house number, similar name);
    distractor records are loners. Uses the pipeline's own blocking, record against record.
    """
    from rapidfuzz import fuzz, process
    from blocking import BlockingConfig, generate_candidates
    from pipeline import _release, prepare
    from records import base_fields
    t0 = time.time()
    s1_raw, r_raw = load_split(split)
    s1b, rb = base_fields(s1_raw), base_fields(r_raw)
    del s1_raw, r_raw
    _, r, stats = prepare(s1b, rb, aliases)
    del s1b, rb
    _release()
    pos = pd.Index(r["entity_id"]).get_indexer(pd.Index(rec_ids).astype(str))
    ok = pos >= 0
    q = r.iloc[pos[ok]].reset_index(drop=True)
    cand = generate_candidates(r, q, stats, BlockingConfig(top_k=4, rel_cut=0.0, fallback_k=2), lambda m: None)
    qi, ti = cand["r_idx"].to_numpy(), cand["s1_idx"].to_numpy()
    keep = ti != pos[ok][qi]                     # not the record itself
    qi, ti = qi[keep], ti[keep]
    name = process.cpdist(q["name_a"].take(qi).tolist(), r["name_a"].take(ti).tolist(), scorer=fuzz.token_set_ratio,
                          workers=-1, dtype=np.float32)
    qa, ta = q["addr_a"].take(qi).tolist(), r["addr_a"].take(ti).tolist()
    addr = process.cpdist(qa, ta, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    addr[[not a or not b for a, b in zip(qa, ta)]] = 0.0
    num = np.fromiter((bool(set(a.split()) & set(b.split())) if a and b else False
                       for a, b in zip(q["addr_nums"].take(qi).tolist(), r["addr_nums"].take(ti).tolist())), bool, len(qi))
    f = pd.DataFrame({"q": qi, "name": name, "addr": addr, "num": num})
    f["strong"] = (f["name"] >= 85) & ((f["addr"] >= 80) | f["num"])
    g = f.groupby("q")
    out = pd.DataFrame(0.0, index=np.arange(len(q)), columns=["twin_name", "twin_addr", "twin_num", "twin_strong"],
                       dtype=np.float32)
    idx = g.size().index.to_numpy()
    out.loc[idx, "twin_name"] = g["name"].max().to_numpy() / 100
    out.loc[idx, "twin_addr"] = g["addr"].max().to_numpy() / 100
    out.loc[idx, "twin_num"] = g["num"].max().astype(np.float32).to_numpy()
    out.loc[idx, "twin_strong"] = g["strong"].max().astype(np.float32).to_numpy()
    out.index = q["entity_id"].tolist()
    _log(f"twin features for {len(out):,} {split} records in {time.time() - t0:.0f}s "
         f"(strong twin: {out['twin_strong'].mean():.0%})")
    del r, q, cand
    _release()
    return out.reindex(pd.Index(rec_ids).astype(str)).fillna(0.0)


TWIN_COLS = ["twin_name", "twin_addr", "twin_num", "twin_strong"]


def combiner_matrix(p, is_best, twins: pd.DataFrame, llm: np.ndarray | None) -> np.ndarray:
    cols = [_logit(p), is_best.astype(float)] + [twins[c].to_numpy() for c in TWIN_COLS]
    if llm is not None:
        has = np.isfinite(llm)
        cols += [has.astype(float), np.where(has, _logit(np.nan_to_num(llm, nan=0.5)), 0.0)]
    return np.column_stack(cols)


def _decide_ids(method, s1c, rc, prob, t, ids, truth, vl):
    return macro_f05(ids[decide(method, s1c, rc, prob, t)], truth, vl)


def calibrate_on_validation(base: dict, judge: Judge, jcfg: JudgeConfig, cfg: V7Config) -> dict:
    """Fit combiners on the v6 validation cache (twins only, twins + LLM) and keep what improves F0.5."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    val = base["val"]
    val_ids = val_ids_of_run(base["bundle"])
    truth = load_pairs()
    truth = truth[truth["s1"].isin(val_ids)]
    s1c, s1u = pd.factorize(val["s1"].astype(str))
    rc, ru = pd.factorize(val["r"].astype(str))
    s1c, rc, s1u, ru = s1c.astype(np.int32), rc.astype(np.int32), np.asarray(s1u, dtype=object), np.asarray(ru, dtype=object)
    p, y = val["p"].to_numpy(np.float32), val["y"].to_numpy()
    ids = pd.DataFrame({"s1": s1u[s1c], "r": ru[rc]})
    vl = list(val_ids)
    method, t0 = base["method"], base["t"]

    rows = rescore_rows(s1c, rc, p, cfg.rescore_lo, cfg.rescore_hi)
    twins = twin_features("train", ru[rc[rows]], base["bundle"]["aliases"])
    llm = np.full(len(rows), np.nan, np.float32)
    hard = hard_pairs(s1c, rc, p, np.zeros(len(p), bool), cfg, cfg.val_budget, t0)
    in_rows = np.isin(rows, hard)
    if judge is not None and in_rows.any():
        hr = rows[in_rows]
        sibs = sibling_lists(s1c, rc, p, hr, cfg.sibling_p)
        S1T, RT = texts("train")
        llm[in_rows] = judge.score(build_prompts(s1u, ru, s1c, rc, hr, sibs, S1T, RT, jcfg), _log)
        del S1T, RT
    is_best = best_assignment(s1c, rc, p)[rows]

    def evaluate(prob):
        best_t, m = t0, _decide_ids(method, s1c, rc, prob, t0, ids, truth, vl)
        if method != "select":
            for t in np.round(np.arange(0.3, 0.96, 0.025), 3):
                mt = _decide_ids(method, s1c, rc, prob, t, ids, truth, vl)
                if mt["f05"] > m["f05"]:
                    best_t, m = t, mt
        return best_t, m

    results = {"none": (t0, _decide_ids(method, s1c, rc, p, t0, ids, truth, vl), None)}
    for name, use_llm in (("twin", False), ("twin+llm", True)):
        if use_llm and not np.isfinite(llm).any():
            continue
        X = combiner_matrix(p[rows], is_best, twins, llm if use_llm else None)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y[rows])
        new = p.copy()
        new[rows] = lr.predict_proba(X)[:, 1]
        t_best, m = evaluate(new)
        results[name] = (t_best, m, lr)
    best = max(results, key=lambda k: results[k][1]["f05"])
    fin = np.isfinite(llm)
    info = {"rescore_rows": int(len(rows)), "llm_rows": int(fin.sum()),
            "auc_base": float(roc_auc_score(y[rows], p[rows])) if len(set(y[rows])) > 1 else None,
            "auc_llm": float(roc_auc_score(y[rows][fin], llm[fin])) if fin.any() and len(set(y[rows][fin])) > 1 else None,
            "auc_twin_strong": float(roc_auc_score(y[rows], twins["twin_strong"])) if len(set(y[rows])) > 1 else None,
            "results": {k: {"threshold": float(v[0]), **v[1]} for k, v in results.items()}, "chosen": best}
    _log(f"validation calibration: {json.dumps(info)}")
    return {"chosen": best, "models": {k: v[2] for k, v in results.items()},
            "thresholds": {k: v[0] for k, v in results.items()}, "info": info}


# ---------------------------------------------------------------------------- run
def run(run_dir: Path, out_dir: Path, artifact_dir: Path, cfg: V7Config = V7Config(),
        jcfg: JudgeConfig = JudgeConfig()) -> dict:
    base = load_base(Path(run_dir))
    _log(f"base run: {len(base['p']):,} candidates, method {base['method']}, threshold {base['t']}, "
         f"validation cache {'yes' if base['val'] is not None else 'no'}")
    if base["val"] is None:
        raise RuntimeError("v7 needs a base run with a validation cache (v6 or later)")
    judge = None
    if cfg.use_llm:
        try:
            judge = Judge(jcfg)
            _log(f"judge loaded: {jcfg.model} on {judge.devices}")
        except Exception as e:  # the twin combiner still runs
            _log(f"!! LLM judge unavailable ({e}); continuing with twin features only")
    calib = calibrate_on_validation(base, judge, jcfg, cfg)

    s1_idx, r_idx, p = base["s1_idx"], base["r_idx"], base["p"]
    french = base["country"][s1_idx] == "France"
    rows = rescore_rows(s1_idx, r_idx, p, cfg.rescore_lo, cfg.rescore_hi)
    _log(f"test rows to re-score: {len(rows):,} ({french[rows].mean():.0%} French)")
    twins = twin_features("test", base["r_ids"][r_idx[rows]], base["bundle"]["aliases"])
    llm = np.full(len(rows), np.nan, np.float32)
    if judge is not None and calib["models"].get("twin+llm") is not None:
        hard = hard_pairs(s1_idx, r_idx, p, french, cfg, cfg.test_budget, base["t"])
        in_rows = np.isin(rows, hard)
        hr = rows[in_rows]
        _log(f"LLM on {len(hr):,} hardest test pairs ({french[hr].mean():.0%} French)")
        sibs = sibling_lists(s1_idx, r_idx, p, hr, cfg.sibling_p)
        S1T, RT = texts("test")
        llm[in_rows] = judge.score(build_prompts(base["s1_ids"], base["r_ids"], s1_idx, r_idx, hr, sibs, S1T, RT, jcfg), _log)
        del S1T, RT
    is_best = best_assignment(s1_idx, r_idx, p)[rows]

    def rescored(name):
        lr = calib["models"].get(name)
        if lr is None:
            return p, base["t"]
        X = combiner_matrix(p[rows], is_best, twins, llm if name == "twin+llm" else None)
        out = p.copy()
        out[rows] = lr.predict_proba(X)[:, 1]
        return out, calib["thresholds"][name]

    chosen = calib["chosen"]
    new, t_new = rescored(chosen)
    _log(f"chosen combination: {chosen}")
    final = decide(base["method"], s1_idx, r_idx, new, t_new)
    base_mask = decide(base["method"], s1_idx, r_idx, p, base["t"])
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    w = lambda path, m: write_id_lists(path, "matched_entity_ids", base["s1_ids"], base["r_ids"], s1_idx[m], r_idx[m])
    w(out_dir / "matching_results.tsv", final)
    write_id_lists(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", base["s1_ids"], base["r_ids"], s1_idx, r_idx)
    w(artifact_dir / "matching_results_base.tsv", base_mask)
    for name in ("twin", "twin+llm"):
        if name != chosen and calib["models"].get(name) is not None:
            prob, t = rescored(name)
            w(artifact_dir / f"matching_results_{name.replace('+', '_')}.tsv", decide(base["method"], s1_idx, r_idx, prob, t))
    fr_only = np.where(french, new, p)   # chosen combination applied to France only
    w(artifact_dir / "matching_results_chosen_france_only.tsv", decide(base["method"], s1_idx, r_idx, fr_only, t_new))
    pd.DataFrame({"row": rows, "p": p[rows], "p_new": new[rows], "llm": llm, "french": french[rows],
                  **{c: twins[c].to_numpy() for c in TWIN_COLS}}).to_parquet(artifact_dir / "v7_scores.parquet", index=False)
    changed = final != base_mask
    summary = {"chosen": chosen, "rescore_rows": int(len(rows)), "llm_rows": int(np.isfinite(llm).sum()),
               "french_share_rescored": float(french[rows].mean()),
               "decisions_changed": int(changed.sum()), "changed_french": int((changed & french).sum()),
               "added": int((final & ~base_mask).sum()), "removed": int((base_mask & ~final).sum()),
               "matches": int(final.sum()), "calibration": calib["info"]}
    (artifact_dir / "v7_report.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"v7 done: {json.dumps(summary, default=str)}")
    return summary
