"""End-to-end pipeline: a training stage (validate + fit) and a test stage, each in its own process.

Validation scenario (built from train, mimicking test):
  * "ghost" S1 entities (ghost_frac) are removed from the S1 table, so their S2/S3
    records become look-alike distractors with no owner (~2.3 distractors per entity,
    the load estimated for test).
  * the remaining S1 entities are split into folds: a validation fold (val_frac), a fit fold
    (fit_frac), and the rest. Aliases and the bi-encoder learn from non-validation pairs.
  * record groups follow the entity folds: records whose candidates touch a validation
    entity are held out; records touching a fit entity fit LightGBM (so every fit entity has
    all of its records, as stage 2's sibling features need); some other records train the
    cross-encoder, so its score is honest when LightGBM sees it.

Models:
  * v2 (GPU): fine-tuned bi-encoder (dense candidates + cosine) and cross-encoder (pair score).
  * stage 1: LightGBM on pair features. Its scores on fit rows are out-of-fold.
  * stage 2 (v6): sibling features (how the record compares with the records already
    assigned to the entity) + a second LightGBM on uncertain pairs.
  * decision: global threshold or per-entity expected-F0.5 set selection, whichever is
    better on validation.
  * France (v3): self-training on French test candidates before stage 2.
Any failure in an optional stage is logged and the run continues without it.

Memory: candidates are integer positions only (no id strings); blocking keys are
generated per chunk; entity ids are mapped only when writing output.
"""

import json
import pickle
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import BlockingConfig, candidate_context, generate_candidates
from data_io import load_pairs, load_split, write_id_lists
from evaluate import blocking_report_idx, error_breakdown, macro_f05
from features import pair_features
from matching import best_threshold, predict, train
from records import KEEP_R, KEEP_S1, apply_aliases, base_fields, compact, country_stats, finalize, learn_aliases
from stage2 import SCOPE_MIN, best_assignment, select_sets, sibling_features, stage2_matrix


@dataclass
class RunConfig:
    ghost_frac: float = 0.19
    val_frac: float = 0.01            # validation entities (their records: ~7% of all)
    fit_frac: float = 0.02            # fit entities (their records: ~13% of all) -> LightGBM training rows
    oof_folds: int = 2                # stage-1 scores on fit rows are out-of-fold (honest input for stage 2)
    test_chunk_rows: int = 6_000_000  # test candidates are featurised and scored in record-aligned chunks
    seed: int = 42
    blocking: BlockingConfig = None
    use_neural: bool = True           # v2 GPU stages (bi-encoder + cross-encoder)
    stage2: bool = True               # v6 sibling stage + set selection
    france_self_train: bool = True    # v3 pseudo-label self-training on French test candidates
    neural: object = None             # neural.NeuralConfig; created when use_neural is set

    def __post_init__(self):
        self.blocking = self.blocking or BlockingConfig()
        if self.use_neural and self.neural is None:
            from neural import NeuralConfig
            self.neural = NeuralConfig()
        if not self.use_neural:
            self.neural = None


class Log:
    """Timestamped log lines with process RAM and free system RAM (to locate memory pressure)."""

    def __init__(self):
        self.t0 = time.time()
        try:
            import psutil
            self.proc, self.vm = psutil.Process(), psutil.virtual_memory
        except ImportError:
            self.proc = None

    def __call__(self, msg):
        mem = ""
        if self.proc is not None and not msg.startswith("  "):
            mem = f" [ram {self.proc.memory_info().rss / 1e9:.1f} GB, free {self.vm().available / 1e9:.1f} GB]"
        print(f"[{time.time() - self.t0:7.0f}s] {msg}{mem}", flush=True)


def _guarded(name, fn, log):
    """Run an optional stage; on any failure log it and return None so the run can continue."""
    try:
        return fn()
    except Exception:
        log(f"!! {name} failed, continuing without it:\n{traceback.format_exc()}")
        return None


def _release():
    """Hand freed memory back: Python garbage and arrow's buffer pool."""
    import gc
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def _free_gpu():
    _release()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def prepare(s1_base, r_base, aliases, chunk: int = 2_000_000):
    """Alias-mapped, finalised, compact record tables. S2/S3 records are processed in chunks so
    the intermediate Python-object columns never exist for all ~10M records at once."""
    s1 = apply_aliases(s1_base, aliases)
    stats = country_stats(s1)
    s1 = compact(finalize(s1, stats), KEEP_S1)
    parts = [compact(finalize(apply_aliases(r_base.iloc[i:i + chunk].reset_index(drop=True), aliases), stats), KEEP_R)
             for i in range(0, len(r_base), chunk)]
    return s1, pd.concat(parts, ignore_index=True), stats


def build_candidates(s1, r, stats, cfg: RunConfig, models: dict, log, key_cand=None):
    cand = key_cand if key_cand is not None else generate_candidates(s1, r, stats, cfg.blocking, log)
    if models.get("bi") is not None:
        from neural import add_dense
        dense = _guarded("dense retrieval", lambda: add_dense(models["bi"], cfg.neural, s1, r, cand, log), log)
        if dense is not None:
            cand = dense
        else:
            models["bi"] = None
    return candidate_context(cand)


def add_ce_scores(ce, s1, r, cand, log):
    from neural import score_in_chunks
    scores = _guarded("cross-encoder scoring", lambda: score_in_chunks(ce, s1, r, cand, log=log), log)
    cand = cand.copy()
    cand["ce_score"] = scores if scores is not None else np.float32(np.nan)
    return cand


def id_pairs(s1, r, s1_idx, r_idx) -> pd.DataFrame:
    """Entity-id pairs for a (small) set of candidate rows — only for evaluation."""
    return pd.DataFrame({"s1": s1["entity_id"].take(s1_idx).array, "r": r["entity_id"].take(r_idx).array})


def decide(method: str, s1_idx, r_idx, p1, p2, t1, t2) -> np.ndarray:
    """Row mask of predicted matches."""
    if method == "select":
        return select_sets(s1_idx, r_idx, p2)
    if method == "stage2":
        return best_assignment(s1_idx, r_idx, p2) & (p2 >= t2)
    return best_assignment(s1_idx, r_idx, p1) & (p1 >= t1)


# ---------------------------------------------------------------------------- training stage
def validate(cfg: RunConfig, log=Log()):
    """Build the validation scenario, fit + evaluate everything, return what the test stage needs."""
    rng = np.random.default_rng(cfg.seed)
    s1_raw, r_raw = load_split("train")
    pairs = load_pairs()
    log(f"train: {len(s1_raw):,} S1, {len(r_raw):,} S2/S3, {len(pairs):,} true pairs")

    ids = s1_raw["entity_id"].astype(str).to_numpy()
    ghost = rng.random(len(ids)) < cfg.ghost_frac
    kept_ids = ids[~ghost]
    truth = pairs[pairs["s1"].isin(set(kept_ids))].reset_index(drop=True)

    s1_base = base_fields(s1_raw[~ghost])
    r_base = base_fields(r_raw)
    del s1_raw, r_raw
    _release()
    # entity folds, by position in the kept S1 table: 1 = validation, 2 = fit, 0 = rest
    fold = np.zeros(len(kept_ids), np.int8)
    u = rng.random(len(kept_ids))
    fold[u < cfg.val_frac] = 1
    fold[(u >= cfg.val_frac) & (u < cfg.val_frac + cfg.fit_frac)] = 2
    val_ids = set(kept_ids[fold == 1])
    log(f"scenario: {ghost.sum():,} ghost S1 removed; {int((fold == 1).sum()):,} validation, "
        f"{int((fold == 2).sum()):,} fit, {int((fold == 0).sum()):,} other entities")

    learn_truth = truth[~truth["s1"].isin(val_ids)].reset_index(drop=True)
    aliases = learn_aliases(s1_base, r_base, learn_truth)
    log(f"aliases: {aliases.summary()}")
    s1, r, stats = prepare(s1_base, r_base, aliases)
    del s1_base, r_base
    _release()
    log(f"records prepared: {len(s1):,} S1, {len(r):,} S2/S3")

    s1_index, r_index = pd.Index(s1["entity_id"]), pd.Index(r["entity_id"])
    t_s1 = s1_index.get_indexer(truth["s1"]).astype(np.int32)
    t_r = r_index.get_indexer(truth["r"]).astype(np.int32)
    owner = np.full(len(r), -1, np.int32)
    owner[t_r] = t_s1

    key_cand = generate_candidates(s1, r, stats, cfg.blocking, log)
    log(f"key candidates: {len(key_cand):,}")

    models = {"bi": None, "ce": None}
    if cfg.neural is not None:
        from neural import record_texts, train_bi_encoder
        bp = rng.choice(np.flatnonzero(fold[t_s1] != 1), min(int((fold[t_s1] != 1).sum()), cfg.neural.bi_train_pairs),
                        replace=False)
        models["bi"] = _guarded("bi-encoder training", lambda: train_bi_encoder(
            cfg.neural, record_texts(s1, t_s1[bp]), record_texts(r, t_r[bp]), log), log)
        del bp
    _free_gpu()
    log("bi-encoder ready" if models["bi"] is not None else "no bi-encoder")

    cand = build_candidates(s1, r, stats, cfg, models, log, key_cand=key_cand)
    del key_cand
    cs1, cr = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    cand["y"] = (owner[cr] == cs1).astype(np.int8)
    is_v = fold[t_s1] == 1
    report = {"all": blocking_report_idx(cs1, cr, t_s1, t_r, len(r)),
              "val": blocking_report_idx(cs1, cr, t_s1[is_v], t_r[is_v], len(r))}
    log(f"blocking: {json.dumps(report)}")

    # record groups from entity folds: 1 = validation, 2 = fit, 3 = cross-encoder training, 0 = unused
    group = np.zeros(len(r), np.int8)
    group[cr[fold[cs1] == 2]] = 2
    group[cr[fold[cs1] == 1]] = 1
    if cfg.neural is not None:
        rest = np.unique(cr[group[cr] == 0])
        group[rng.choice(rest, min(len(rest), cfg.neural.ce_train_records), replace=False)] = 3
        from neural import CrossEncoder, pair_texts

        def fit_ce():
            rows = cand[group[cr] == 3]
            if len(rows) > cfg.neural.ce_max_pairs:
                rows = rows.sample(cfg.neural.ce_max_pairs, random_state=cfg.seed)
            log(f"cross-encoder training pairs: {len(rows):,} ({rows['y'].mean():.1%} positive)")
            ce = CrossEncoder(cfg.neural)
            a, b = pair_texts(s1, r, rows)
            ce.fit(a, b, rows["y"].to_numpy(), log)
            return ce
        models["ce"] = _guarded("cross-encoder training", fit_ce, log)

    use = cand[np.isin(group[cr], (1, 2))].reset_index(drop=True)
    del cand, cs1, cr
    _release()
    us1, ur = use["s1_idx"].to_numpy(), use["r_idx"].to_numpy()
    is_val = group[ur] == 1
    y = use["y"].to_numpy()
    log(f"feature rows: {len(use):,} ({(~is_val).sum():,} fit, {is_val.sum():,} validation)")
    if models["ce"] is not None:
        use = add_ce_scores(models["ce"], s1, r, use, log)
    X = pair_features(s1, r, use, stats, aliases.known_states, log)

    # ---- stage 1: early-stopped on validation; out-of-fold scores on fit rows
    m1 = train(X[~is_val], y[~is_val], X[is_val], y[is_val])
    p1 = np.empty(len(use), np.float32)
    p1[is_val] = m1.predict(X[is_val], num_iteration=m1.best_iteration)
    fit_rows = np.flatnonzero(~is_val)
    part = (ur[fit_rows].astype(np.int64) * 2654435761 % 2**32) % cfg.oof_folds
    for k in range(cfg.oof_folds):
        tr, te = fit_rows[part != k], fit_rows[part == k]
        mk = train(X.iloc[tr], y[tr], rounds=max(50, m1.best_iteration), log_every=10_000)
        p1[te] = mk.predict(X.iloc[te])
        del mk
    log(f"stage 1 done (best iteration {m1.best_iteration}, out-of-fold scores on {len(fit_rows):,} fit rows)")

    v_s1, v_r = us1[is_val], ur[is_val]
    val_pairs = id_pairs(s1, r, v_s1, v_r)
    val_list = list(val_ids)
    t1, table = best_threshold(val_pairs, p1[is_val], truth, val_list)
    results = {"stage1": macro_f05(val_pairs[decide("stage1", v_s1, v_r, p1[is_val], None, t1, None)], truth, val_list)}
    log(f"validation stage 1: threshold {t1:.3f} -> {json.dumps(results['stage1'])}")

    # ---- stage 2: sibling features, second model, set selection
    m2, t2, p2 = None, None, p1.copy()
    if cfg.stage2:
        def fit_stage2():
            scope = p1 >= SCOPE_MIN
            sib = pd.DataFrame(index=range(len(use)), columns=[], dtype=np.float32)
            parts = []
            for m in (is_val, ~is_val):   # siblings from each group's own assignment
                rows = np.flatnonzero(m)
                f = sibling_features(us1[rows], ur[rows], p1[rows], scope[rows], r)
                f.index = rows
                parts.append(f)
            sib = pd.concat(parts).sort_index()
            X2 = stage2_matrix(X, p1, sib, ur)
            ent_fold = fold[us1]
            fit2 = (~is_val) & scope & (ent_fold == 2)      # entities whose records are all present
            val2 = is_val & scope & (ent_fold == 1)
            model2 = train(X2[fit2], y[fit2], X2[val2], y[val2])
            out = p1.copy()
            out[val2] = model2.predict(X2[val2], num_iteration=model2.best_iteration)
            return model2, out
        res = _guarded("stage 2", fit_stage2, log)
        if res is not None:
            m2, p2 = res
            t2, _ = best_threshold(val_pairs, p2[is_val], truth, val_list)
            results["stage2"] = macro_f05(val_pairs[decide("stage2", v_s1, v_r, None, p2[is_val], None, t2)],
                                          truth, val_list)
            results["select"] = macro_f05(val_pairs[decide("select", v_s1, v_r, None, p2[is_val], None, None)],
                                          truth, val_list)
            log(f"validation stage 2: threshold {t2:.3f} -> {json.dumps(results['stage2'])}")
            log(f"validation set selection: {json.dumps(results['select'])}")
    method = max(results, key=lambda k: results[k]["f05"])
    log(f"decision method: {method} (F0.5 {results[method]['f05']:.5f})")

    final_mask = decide(method, v_s1, v_r, p1[is_val], p2[is_val], t1, t2)
    country_of = dict(zip(s1["entity_id"].tolist(), s1["country"].tolist()))
    diagnostics = error_breakdown(val_pairs[final_mask], truth, pairs, set(ids[ghost]), val_list, country_of,
                                  val_pairs["r"].to_numpy())
    log(f"diagnostics: {json.dumps(diagnostics)}")
    importance = pd.Series(m1.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    val_cache = val_pairs.assign(y=y[is_val], p1=p1[is_val], p2=p2[is_val])
    return {"aliases": aliases, "models": models, "m1": m1, "m2": m2, "t1": t1, "t2": t2, "method": method,
            "results": results, "threshold_table": table, "blocking": report, "importance": importance,
            "diagnostics": diagnostics, "val_cache": val_cache,
            "neural": {k: v is not None for k, v in models.items()}}


# ---------------------------------------------------------------------------- test stage
def predict_test(cfg: RunConfig, b: dict, models: dict, m1, m2, out_dir: Path, artifact_dir: Path,
                 log=Log()) -> dict:
    s1_raw, r_raw = load_split("test")
    log(f"test: {len(s1_raw):,} S1, {len(r_raw):,} S2/S3")
    s1_base, r_base = base_fields(s1_raw), base_fields(r_raw)
    del s1_raw, r_raw
    _release()
    s1, r, stats = prepare(s1_base, r_base, b["aliases"])
    del s1_base, r_base
    _release()
    log(f"test records prepared: {len(s1):,} S1, {len(r):,} S2/S3")
    cand = build_candidates(s1, r, stats, cfg, models, log)
    cand = cand.iloc[np.argsort(cand["r_idx"].to_numpy(), kind="stable")].reset_index(drop=True)
    cs1, cr = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    log(f"test candidates: {len(cand):,} ({len(cand) / len(r):.2f}/record)")

    # stage 1 in record-aligned chunks; keep only stage-2 inputs for in-scope rows
    p1 = np.empty(len(cand), np.float32)
    keep_base, french = [], []
    is_fr_s1 = s1["country"].to_numpy() == "France"
    start = 0
    while start < len(cand):
        stop = min(start + cfg.test_chunk_rows, len(cand))
        while stop < len(cand) and cr[stop] == cr[stop - 1]:
            stop += 1
        chunk = cand.iloc[start:stop].reset_index(drop=True)
        if models.get("ce") is not None:
            chunk = add_ce_scores(models["ce"], s1, r, chunk, log)
        X = pair_features(s1, r, chunk, stats, b["aliases"].known_states, lambda m: None)
        p = predict(m1, X).astype(np.float32)
        p1[start:stop] = p
        pos = np.arange(start, stop)
        if m2 is not None:
            sc = p >= SCOPE_MIN
            keep_base.append((pos[sc], X[sc].reset_index(drop=True)))
        if cfg.france_self_train:
            fr = is_fr_s1[chunk["s1_idx"].to_numpy()]
            if fr.any():
                french.append((pos[fr], X[fr].reset_index(drop=True)))
        log(f"  test stage 1: {stop:,}/{len(cand):,} candidates")
        del X, chunk
        start = stop
    p1_base = p1.copy()

    france_info = None
    if french:
        from france import FranceConfig, self_train
        pos = np.concatenate([p for p, _ in french])
        Xf = pd.concat([x for _, x in french], ignore_index=True)
        del french
        res = _guarded("france self-training", lambda: self_train(
            s1, r, cand.iloc[pos].reset_index(drop=True), Xf, p1[pos], m1, b["aliases"].known_states,
            FranceConfig(), log), log)
        del Xf
        if res is not None:
            new, france_info = res
            france_info.pop("_state_only_prob", None)
            p1[pos] = new

    def stage2_scores(p_in):
        if m2 is None:
            return p_in
        pos = np.concatenate([p for p, _ in keep_base])
        base = pd.concat([x for _, x in keep_base], ignore_index=True)
        scope = np.zeros(len(p_in), bool)
        scope[pos] = p_in[pos] >= SCOPE_MIN
        sib = sibling_features(cs1, cr, p_in, scope, r)
        sel = scope[pos]
        X2 = stage2_matrix(base[sel].reset_index(drop=True), p_in[pos[sel]], sib.iloc[pos[sel]], cr[pos[sel]])
        out = p_in.copy()
        out[pos[sel]] = predict(m2, X2).astype(np.float32)
        return out

    p2 = _guarded("test stage 2", lambda: stage2_scores(p1), log)
    method = b["method"]
    if p2 is None:
        p2, method = p1, "stage1"
    log(f"test decision: {method}")
    final = decide(method, cs1, cr, p1, p2, b["t1"], b["t2"])

    s1_ids, r_ids = s1["entity_id"].tolist(), r["entity_id"].tolist()
    write_id_lists(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, r_ids, cs1, cr)
    write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, r_ids, cs1[final], cr[final])
    # variants for leaderboard comparisons (same run)
    s1_only = decide("stage1", cs1, cr, p1_base, None, b["t1"], None)
    write_id_lists(artifact_dir / "matching_results_stage1.tsv", "matched_entity_ids", s1_ids, r_ids,
                   cs1[s1_only], cr[s1_only])
    if france_info is not None:
        p2_nofr = _guarded("test stage 2 without France", lambda: stage2_scores(p1_base), log)
        if p2_nofr is not None:
            nofr = decide(method, cs1, cr, p1_base, p2_nofr, b["t1"], b["t2"])
            write_id_lists(artifact_dir / "matching_results_no_france_st.tsv", "matched_entity_ids", s1_ids,
                           r_ids, cs1[nofr], cr[nofr])
    # cached scores: thresholds / decisions can be re-tuned without re-running
    pd.DataFrame({"s1_idx": cs1, "r_idx": cr, "p1_base": p1_base, "p1": p1, "p2": p2}).to_parquet(
        artifact_dir / "test_scores.parquet", index=False)
    pd.DataFrame({"entity_id": s1_ids, "country": s1["country"].tolist()}).to_parquet(artifact_dir / "test_s1_ids.parquet")
    pd.DataFrame({"entity_id": r_ids}).to_parquet(artifact_dir / "test_r_ids.parquet")

    ctry = s1["country"].to_numpy()[cs1[final]]
    summary = {"s1": len(s1_ids), "candidates": int(len(cand)), "matches": int(final.sum()), "method": method,
               "s1_with_matches": int(len(np.unique(cs1[final]))),
               "per_country_matches": pd.Series(ctry).value_counts().to_dict(),
               "france_self_training": france_info}
    log(f"test written to {out_dir}: {json.dumps(summary, default=str)}")
    return summary


# ---------------------------------------------------------------------------- stages as processes
def train_stage(out_dir: Path, artifact_dir: Path) -> dict:
    """Validation + model fitting from a pickled RunConfig; saves report and bundle (own process)."""
    with open(artifact_dir / "run_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    log = Log()
    v = validate(cfg, log)
    v["m1"].save_model(str(artifact_dir / "model.txt"))
    if v["m2"] is not None:
        v["m2"].save_model(str(artifact_dir / "model_stage2.txt"))
    v["threshold_table"].to_csv(artifact_dir / "threshold_table.tsv", sep="\t", index=False)
    v["val_cache"].to_parquet(artifact_dir / "val_scores.parquet", index=False)
    if v["models"].get("bi") is not None:
        v["models"]["bi"].save(str(artifact_dir / "bi_encoder"))
    if v["models"].get("ce") is not None:
        v["models"]["ce"].save(artifact_dir / "cross_encoder")
    with open(artifact_dir / "bundle.pkl", "wb") as f:
        pickle.dump({"cfg": cfg, "aliases": v["aliases"], "t1": v["t1"], "t2": v["t2"], "method": v["method"],
                     "has_bi": v["models"].get("bi") is not None, "has_ce": v["models"].get("ce") is not None,
                     "has_m2": v["m2"] is not None}, f)
    report = {"config": asdict(cfg), "validation": v["results"], "method": v["method"], "t1": v["t1"], "t2": v["t2"],
              "blocking": v["blocking"], "neural": v["neural"], "diagnostics": v["diagnostics"],
              "top_features": v["importance"].head(30).round(1).to_dict()}
    (artifact_dir / "run_report.json").write_text(json.dumps(report, indent=2, default=str))
    log("training stage saved")
    return report


def test_stage(out_dir: Path, artifact_dir: Path) -> dict:
    """Test phase from a saved bundle; runs in its own process (see `run`)."""
    import lightgbm as lgb
    log = Log()
    with open(artifact_dir / "bundle.pkl", "rb") as f:
        b = pickle.load(f)
    models = {"bi": None, "ce": None}
    if b["has_bi"] or b["has_ce"]:
        from neural import CrossEncoder, load_bi_encoder
        if b["has_bi"]:
            models["bi"] = load_bi_encoder(artifact_dir / "bi_encoder")
        if b["has_ce"]:
            models["ce"] = CrossEncoder(b["cfg"].neural, path=artifact_dir / "cross_encoder")
    m1 = lgb.Booster(model_file=str(artifact_dir / "model.txt"))
    m2 = lgb.Booster(model_file=str(artifact_dir / "model_stage2.txt")) if b["has_m2"] else None
    log(f"test stage: bundle loaded (bi {b['has_bi']}, cross {b['has_ce']}, stage 2 {b['has_m2']}, "
        f"method {b['method']})")
    summary = predict_test(b["cfg"], b, models, m1, m2, out_dir, artifact_dir, log)
    (artifact_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _run_stage(stage: str, out_dir: Path, artifact_dir: Path):
    """Run a stage function of this module in a fresh Python process and stream its log."""
    import subprocess
    import sys
    code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
            f"from pathlib import Path; import pipeline; "
            f"pipeline.{stage}(Path({str(out_dir)!r}), Path({str(artifact_dir)!r}))")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        if "Loading weights" not in line and "Writing model shards" not in line:
            print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"{stage} failed with exit code {proc.returncode} (see log above)")


def run(cfg: RunConfig, out_dir: Path, artifact_dir: Path, log=Log()) -> dict:
    """Training and test stages, each in its own process, so neither inherits the other's memory
    (the calling process only holds the config and the reports)."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "run_config.pkl", "wb") as f:
        pickle.dump(cfg, f)
    log("training stage (own process)")
    _run_stage("train_stage", out_dir, artifact_dir)
    log("test stage (own process)")
    _run_stage("test_stage", out_dir, artifact_dir)
    report = json.loads((artifact_dir / "run_report.json").read_text())
    report["test"] = json.loads((artifact_dir / "test_summary.json").read_text())
    (artifact_dir / "run_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
