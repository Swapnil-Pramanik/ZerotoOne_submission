"""v7: an LLM judge on the hardest pairs of a finished run (post-processing, no retraining).

Input: the artifacts of a finished v5 or v6 run (cached test scores, id tables, decision
settings; v6 also has cached validation scores). Steps:

1. Hard pairs: each record's best candidate with an uncertain probability, plus a close
   second candidate. France gets a wider band and a reserved share of the budget (it has
   no training labels, so the base model is least reliable there).
2. Prompts: the S1 business, up to three records confidently assigned to it (siblings), and
   the candidate record, from the raw dataset text (native scripts kept).
3. LLM judge scores (llm_judge.py).
4. Combination, calibrated on labelled pairs:
   * v6 base: the same hard-pair selection on its validation scores, a logistic regression
     y ~ [logit p, logit llm, best-candidate flag], and the decision re-tuned; applied only
     if validation macro F0.5 improves.
   * v5 base (no validation cache): the LLM may only flip a decision where it is very sure
     (P(Yes) >= 0.97 to include, <= 0.03 to exclude); these cut-offs are checked against the
     probe's labelled pairs (notebooks/10_llm_probe) before use.
5. Output: the new submission plus the base decision, and the LLM scores for later use.
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
    test_budget: int = 150_000       # hard test pairs sent to the LLM
    val_budget: int = 40_000         # hard validation pairs for calibration (v6 base)
    france_share: float = 0.4        # share of the test budget reserved for French pairs
    lo: float = 0.05                 # uncertainty band on the base probability ...
    hi: float = 0.95
    fr_lo: float = 0.02              # ... wider for France
    fr_hi: float = 0.985
    sibling_p: float = 0.9           # siblings shown to the LLM: records assigned to the entity with p >= this
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


def calibrate_on_validation(base: dict, judge: Judge, jcfg: JudgeConfig, cfg: V7Config) -> dict:
    """Fit the p/LLM combination on the v6 validation cache and check it improves validation F0.5."""
    from sklearn.linear_model import LogisticRegression
    val = base["val"]
    val_ids = val_ids_of_run(base["bundle"])
    truth = load_pairs()
    truth = truth[truth["s1"].isin(val_ids)]
    s1c, s1u = pd.factorize(val["s1"].astype(str))
    rc, ru = pd.factorize(val["r"].astype(str))
    s1c, rc = s1c.astype(np.int32), rc.astype(np.int32)
    p, y = val["p"].to_numpy(np.float32), val["y"].to_numpy()
    french = np.zeros(len(p), bool)
    rows = hard_pairs(s1c, rc, p, french, cfg, cfg.val_budget, base["t"])
    sibs = sibling_lists(s1c, rc, p, rows, cfg.sibling_p)
    S1T, RT = texts("train")
    llm = judge.score(build_prompts(np.asarray(s1u), np.asarray(ru), s1c, rc, rows, sibs, S1T, RT, jcfg), _log)
    is_best = best_assignment(s1c, rc, p)[rows]
    Xc = np.column_stack([_logit(p[rows]), _logit(llm), is_best.astype(float)])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xc, y[rows])
    new = p.copy()
    new[rows] = lr.predict_proba(Xc)[:, 1]
    ids = pd.DataFrame({"s1": np.asarray(s1u)[s1c], "r": np.asarray(ru)[rc]})
    vl = list(val_ids)
    before = macro_f05(ids[decide(base["method"], s1c, rc, p, base["t"])], truth, vl)
    # re-tune the threshold after the change (set selection needs none)
    best_t, after = base["t"], macro_f05(ids[decide(base["method"], s1c, rc, new, base["t"])], truth, vl)
    if base["method"] != "select":
        for t in np.round(np.arange(0.3, 0.96, 0.025), 3):
            m = macro_f05(ids[decide(base["method"], s1c, rc, new, t)], truth, vl)
            if m["f05"] > after["f05"]:
                best_t, after = t, m
    from sklearn.metrics import roc_auc_score
    auc_llm = float(roc_auc_score(y[rows], llm)) if len(set(y[rows])) > 1 else None
    auc_p = float(roc_auc_score(y[rows], p[rows])) if len(set(y[rows])) > 1 else None
    info = {"hard_val_pairs": int(len(rows)), "auc_llm": auc_llm, "auc_base": auc_p, "coef": lr.coef_.round(3).tolist(),
            "before": before, "after": after, "threshold": float(best_t)}
    _log(f"validation calibration: {json.dumps(info)}")
    return {"model": lr, "threshold": float(best_t), "apply": after["f05"] > before["f05"], "info": info}


# ---------------------------------------------------------------------------- run
def run(run_dir: Path, out_dir: Path, artifact_dir: Path, cfg: V7Config = V7Config(),
        jcfg: JudgeConfig = JudgeConfig()) -> dict:
    base = load_base(Path(run_dir))
    _log(f"base run: {len(base['p']):,} candidates, method {base['method']}, threshold {base['t']}, "
         f"validation cache {'yes' if base['val'] is not None else 'no'}")
    judge = Judge(jcfg)
    _log(f"judge loaded: {jcfg.model} on {judge.devices}")

    calib = None
    if base["val"] is not None:
        calib = calibrate_on_validation(base, judge, jcfg, cfg)

    s1_idx, r_idx, p = base["s1_idx"], base["r_idx"], base["p"]
    french = base["country"][s1_idx] == "France"
    rows = hard_pairs(s1_idx, r_idx, p, french, cfg, cfg.test_budget, base["t"])
    _log(f"hard test pairs: {len(rows):,} ({french[rows].mean():.0%} French)")
    sibs = sibling_lists(s1_idx, r_idx, p, rows, cfg.sibling_p)
    S1T, RT = texts("test")
    llm = judge.score(build_prompts(base["s1_ids"], base["r_ids"], s1_idx, r_idx, rows, sibs, S1T, RT, jcfg), _log)

    new, t_new, mode = p.copy(), base["t"], "none"
    if calib is not None and calib["apply"]:
        Xc = np.column_stack([_logit(p[rows]), _logit(llm), best_assignment(s1_idx, r_idx, p)[rows].astype(float)])
        new[rows] = calib["model"].predict_proba(Xc)[:, 1]
        t_new, mode = calib["threshold"], "validation-calibrated"
    elif calib is None:
        # no validation cache: only flip where the LLM is very sure (cut-offs from the probe on labelled pairs)
        inc, exc = (llm >= 0.97), (llm <= 0.03)
        new[rows[inc]] = np.maximum(p[rows[inc]], base["t"] + 1e-3)
        new[rows[exc]] = np.minimum(p[rows[exc]], base["t"] - 1e-3)
        mode = "confident-flips"
    _log(f"combination mode: {mode}")

    final = decide(base["method"], s1_idx, r_idx, new, t_new)
    base_mask = decide(base["method"], s1_idx, r_idx, p, base["t"])
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", base["s1_ids"], base["r_ids"],
                   s1_idx[final], r_idx[final])
    write_id_lists(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", base["s1_ids"], base["r_ids"], s1_idx, r_idx)
    write_id_lists(artifact_dir / "matching_results_base.tsv", "matched_entity_ids", base["s1_ids"], base["r_ids"],
                   s1_idx[base_mask], r_idx[base_mask])
    # France-only variant: LLM changes applied to French pairs only
    fr_only = np.where(french, new, p)
    fr_mask = decide(base["method"], s1_idx, r_idx, fr_only, t_new if mode == "validation-calibrated" else base["t"])
    write_id_lists(artifact_dir / "matching_results_llm_france_only.tsv", "matched_entity_ids", base["s1_ids"],
                   base["r_ids"], s1_idx[fr_mask], r_idx[fr_mask])
    pd.DataFrame({"row": rows, "llm": llm, "p": p[rows], "p_new": new[rows], "french": french[rows]}).to_parquet(
        artifact_dir / "llm_scores.parquet", index=False)
    changed = final != base_mask
    summary = {"mode": mode, "hard_pairs": int(len(rows)), "french_share": float(french[rows].mean()),
               "decisions_changed": int(changed.sum()), "changed_french": int((changed & french).sum()),
               "added": int((final & ~base_mask).sum()), "removed": int((base_mask & ~final).sum()),
               "matches": int(final.sum()), "calibration": calib["info"] if calib else None}
    (artifact_dir / "v7_report.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"v7 done: {json.dumps(summary, default=str)}")
    return summary
