"""v3: self-training for France, the test-only country with no training labels.

After the base model has scored every test candidate:

1. Pseudo-labels on French candidates. Positives are records whose single best S1 entity
   scores >= pos_prob. Negatives are the other candidates of those records (each record
   has at most one owner) plus any pair scoring <= neg_prob.
2. French state aliases are learned from the pseudo-positive pairs: which départements and
   cities of S2/S3 addresses consistently go with which S1 région. The state-agreement
   feature, NaN for France until now, then carries signal. Only the organisers' unlabelled
   test data is used.
3. A France-only LightGBM is fitted on the pseudo-labelled rows. France's final
   probability blends it with the base model (blend = weight of the France model), which
   limits self-confirmation. The base model re-scored with only the new state feature is
   returned too ("state-only" variant), so both steps can be compared on the leaderboard.
"""

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from aliases import learn_states
from matching import PARAMS, predict


@dataclass
class FranceConfig:
    country: str = "France"
    pos_prob: float = 0.97
    neg_prob: float = 0.03
    min_pseudo_pos: int = 2_000
    rounds: int = 300
    blend: float = 0.5


def pseudo_labels(cand: pd.DataFrame, prob: np.ndarray, cfg: FranceConfig) -> np.ndarray:
    """1 / 0 for confident rows, -1 for rows left out. cand needs r_idx."""
    df = pd.DataFrame({"r": cand["r_idx"].to_numpy(), "p": prob})
    best = df.groupby("r")["p"].transform("max").to_numpy()
    is_best = prob == best
    confident_record = best >= cfg.pos_prob
    y = np.full(len(df), -1, dtype=np.int8)
    y[confident_record & ~is_best] = 0
    y[prob <= cfg.neg_prob] = 0
    y[confident_record & is_best] = 1
    return y


def state_match_column(s1_state, r_state, known: set, mapping: dict) -> np.ndarray:
    """Recompute the state-agreement feature with extra (French) state aliases."""
    out = np.full(len(s1_state), np.nan, dtype=np.float32)
    for i, (a, rs) in enumerate(zip(s1_state, r_state)):
        parts = [mapping.get(p, p) for p in rs.split("|")] if rs else []
        if a in known and any(p in known for p in parts):
            out[i] = float(a in parts)
    return out


def self_train(s1: pd.DataFrame, r: pd.DataFrame, cand: pd.DataFrame, X: pd.DataFrame, prob: np.ndarray,
               base_model, known_states: set, cfg: FranceConfig = FranceConfig(), log=print):
    """Return (new French probabilities, info dict). All inputs are the French rows only."""
    t0 = time.time()
    y = pseudo_labels(cand, prob, cfg)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    info = {"pseudo_pos": n_pos, "pseudo_neg": n_neg, "rows": int(len(y))}
    if n_pos < cfg.min_pseudo_pos:
        log(f"  france: only {n_pos:,} pseudo-positives, keeping base probabilities")
        return prob, info

    # French state aliases from pseudo-positive pairs
    s1_state = np.asarray(s1["state_s1"].take(cand["s1_idx"].to_numpy()).tolist(), dtype=object)
    r_state = np.asarray(r["state_r"].take(cand["r_idx"].to_numpy()).tolist(), dtype=object)
    pos = y == 1
    mapping, known = learn_states(s1_state[pos].tolist(), [s.split("|") if s else [] for s in r_state[pos]],
                                  min_count=30, min_share=0.9)
    known = known | set(known_states)
    X = X.copy()
    X["state_match"] = state_match_column(s1_state, r_state, known, mapping)
    info.update(state_aliases=len(mapping), known_states=len(known),
                state_match_coverage=float(np.isfinite(X["state_match"]).mean()))
    log(f"  france: {n_pos:,} pseudo-positives, {n_neg:,} pseudo-negatives; {len(mapping):,} state aliases, "
        f"state feature now set on {info['state_match_coverage']:.0%} of French pairs")

    # France model on pseudo-labels, blended with the (re-scored) base model
    lab = y >= 0
    fr_model = lgb.train({**PARAMS, "learning_rate": 0.05}, lgb.Dataset(X[lab], y[lab]), num_boost_round=cfg.rounds)
    base = predict(base_model, X).astype(np.float32)   # base model, now with the French state feature
    fr = fr_model.predict(X)
    new = ((1 - cfg.blend) * base + cfg.blend * fr).astype(np.float32)
    info.update(mean_prob_before=float(prob.mean()), mean_prob_state_only=float(base.mean()),
                mean_prob_after=float(new.mean()))
    info["_state_only_prob"] = base
    log(f"  france self-training done in {time.time() - t0:.0f}s")
    return new, info
