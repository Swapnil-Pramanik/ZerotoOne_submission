"""Official metric (macro F0.5 over S1 entities, singletons included) and blocking diagnostics."""

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> dict:
    """pred, truth: DataFrames with columns s1, r. Averages F0.5 over every id in s1_ids.

    Per entity: both empty -> 1; exactly one empty -> 0; else F0.5 of the two sets.
    """
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)))
    pred = pred[pred["s1"].isin(s1_ids)].drop_duplicates(["s1", "r"])
    truth = truth[truth["s1"].isin(s1_ids)].drop_duplicates(["s1", "r"])
    tp = pred.merge(truth, on=["s1", "r"]).groupby("s1").size()
    n_pred = pred.groupby("s1").size()
    n_true = truth.groupby("s1").size()
    df = pd.DataFrame(index=s1_ids)
    df["tp"] = tp.reindex(s1_ids).fillna(0).to_numpy()
    df["pred"] = n_pred.reindex(s1_ids).fillna(0).to_numpy()
    df["true"] = n_true.reindex(s1_ids).fillna(0).to_numpy()
    p = np.divide(df["tp"], df["pred"], out=np.zeros(len(df)), where=df["pred"] > 0)
    r = np.divide(df["tp"], df["true"], out=np.zeros(len(df)), where=df["true"] > 0)
    denom = BETA2 * p + r
    f = np.divide((1 + BETA2) * p * r, denom, out=np.zeros(len(df)), where=denom > 0)
    f = np.where((df["pred"] == 0) & (df["true"] == 0), 1.0, f)
    singles = df["true"] == 0
    return {
        "f05": float(f.mean()),
        "precision": float(p[df["pred"] > 0].mean()) if (df["pred"] > 0).any() else float("nan"),
        "recall": float(r[df["true"] > 0].mean()) if (df["true"] > 0).any() else float("nan"),
        "f05_singletons": float(f[singles].mean()) if singles.any() else float("nan"),
        "f05_matched": float(f[~singles].mean()) if (~singles).any() else float("nan"),
        "entities": int(len(df)),
    }


def blocking_report(cand_pairs: pd.DataFrame, truth: pd.DataFrame, s1_ids, n_records: int) -> dict:
    """Pair recall of the candidate set restricted to s1_ids, and candidates per record."""
    truth = truth[truth["s1"].isin(set(s1_ids))]
    hit = truth.merge(cand_pairs[["s1", "r"]].drop_duplicates(), on=["s1", "r"]).shape[0]
    return {"pair_recall": hit / max(len(truth), 1), "true_pairs": int(len(truth)),
            "candidates": int(len(cand_pairs)), "per_record": len(cand_pairs) / max(n_records, 1)}


def error_breakdown(pred: pd.DataFrame, truth: pd.DataFrame, all_pairs: pd.DataFrame, ghost_s1: set,
                    s1_ids, country_of: dict, candidate_records) -> dict:
    """Where do false merges come from, and how does F0.5 split by country?

    Every wrongly matched record is one of: a *real distractor* (owned by no S1 entity in
    the data), a *ghost record* (owned by an S1 entity we removed to add distractors), or a
    record that belongs to *another* S1 entity. Rates are per candidate record of that kind,
    so real vs ghost distractors can be compared (are our synthetic distractors too easy?).
    """
    s1_ids = list(s1_ids)
    owner = dict(zip(all_pairs["r"], all_pairs["s1"]))

    def kind(r):
        o = owner.get(r)
        return "real_distractor" if o is None else ("ghost_record" if o in ghost_s1 else "other_entity_record")

    pred = pred[pred["s1"].isin(set(s1_ids))]
    true_set = set(zip(truth["s1"], truth["r"]))
    wrong = pred[[(a, b) not in true_set for a, b in zip(pred["s1"], pred["r"])]]
    wrong_kind = pd.Series([kind(r) for r in wrong["r"]]).value_counts()
    base = pd.Series([kind(r) for r in pd.unique(np.asarray(candidate_records))]).value_counts()
    out = {"false_merges": int(len(wrong)), "false_merges_by_kind": wrong_kind.to_dict(),
           "candidate_records_by_kind": base.to_dict(),
           "false_merge_rate_by_kind": (wrong_kind / base).fillna(0).round(5).to_dict(), "by_country": {}}
    countries = pd.Series([country_of.get(i) for i in s1_ids], index=s1_ids)
    for c, ids in countries.groupby(countries).groups.items():
        out["by_country"][c] = {k: round(v, 5) for k, v in macro_f05(pred, truth, list(ids)).items()}
    return out


def blocking_report_idx(cand_s1: np.ndarray, cand_r: np.ndarray, true_s1: np.ndarray, true_r: np.ndarray,
                        n_r: int) -> dict:
    """Pair recall of candidates over the given true pairs, all as integer positions."""
    ck = cand_s1.astype(np.int64) * n_r + cand_r
    tk = true_s1.astype(np.int64) * n_r + true_r
    hit = np.isin(tk, ck).sum()
    return {"pair_recall": float(hit / max(len(tk), 1)), "true_pairs": int(len(tk)),
            "candidates": int(len(ck)), "per_record": len(ck) / max(n_r, 1)}
