"""LightGBM pair model, one-owner assignment, and threshold selection."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluate import macro_f05

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
    "seed": 42,
}


def train(X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None, rounds=2000, log_every=100) -> lgb.Booster:
    dtrain = lgb.Dataset(X, y, free_raw_data=True)
    valid = [lgb.Dataset(X_val, y_val, reference=dtrain)] if X_val is not None else []
    callbacks = [lgb.log_evaluation(log_every)]
    if valid:
        callbacks.append(lgb.early_stopping(100, verbose=True))
    return lgb.train(PARAMS, dtrain, num_boost_round=rounds, valid_sets=valid, callbacks=callbacks)


def assign(cand: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    """Each S2/S3 record goes to its single best S1 entity if that probability >= threshold.

    cand needs columns s1, r (entity ids). Returns matched pairs s1, r.
    """
    df = pd.DataFrame({"s1": cand["s1"].array, "r": cand["r"].array, "p": prob})
    best = df.sort_values("p", ascending=False, kind="stable").drop_duplicates("r")
    return best.loc[best["p"] >= threshold, ["s1", "r"]].reset_index(drop=True)


def best_threshold(cand: pd.DataFrame, prob: np.ndarray, truth: pd.DataFrame, s1_ids,
                   grid=np.round(np.arange(0.30, 0.96, 0.025), 3)) -> tuple[float, pd.DataFrame]:
    rows = []
    for t in grid:
        m = macro_f05(assign(cand, prob, t), truth, s1_ids)
        rows.append({"threshold": t, **m})
    table = pd.DataFrame(rows)
    return float(table.loc[table["f05"].idxmax(), "threshold"]), table


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict with columns aligned by name (a feature missing at test time becomes NaN, never a shifted column)."""
    return model.predict(X.reindex(columns=model.feature_name()))
