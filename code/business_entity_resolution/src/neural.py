"""GPU models (v2): a fine-tuned bi-encoder and a fine-tuned cross-encoder.

Both start from a small multilingual transformer (Apache-2.0,
paraphrase-multilingual-MiniLM-L12-v2) and read the *raw* record text, so native
scripts (Devanagari, Tamil, ...) are seen as written, not through anyascii.

* Bi-encoder: contrastive fine-tuning on true (S1, S2/S3) pairs with in-batch
  negatives. Used for dense retrieval (top-k S1 per record, per country, exact
  GPU matmul) and a cosine feature on every candidate pair.
* Cross-encoder: reads "S1 text [SEP] record text" jointly; fine-tuned on
  candidate pairs of records held out from the LightGBM fit (hard negatives);
  its probability becomes a LightGBM feature.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class NeuralConfig:
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # bi-encoder
    bi_train_pairs: int = 400_000
    bi_batch: int = 256
    bi_lr: float = 5e-5
    bi_max_len: int = 64
    dense_k: int = 5
    keep_by_cos: int = 5              # after the union keep a record's top candidates by bi-encoder cosine ...
    keep_by_block: int = 3            # ... plus its top key-blocking candidates ...
    max_cands: int = 8                # ... at most this many per record (bounds cross-encoder and feature cost)
    encode_batch: int = 1024
    query_chunk: int = 512
    # cross-encoder
    ce_train_records: int = 300_000   # records (never used by LightGBM) whose candidates train the cross-encoder
    ce_max_pairs: int = 1_000_000
    ce_top: int = 2                   # cross-encoder scores only each record's top candidates by cosine (others NaN)
    ce_batch: int = 128
    ce_lr: float = 3e-5
    ce_max_len: int = 96
    score_batch: int = 1024


def device() -> str:
    if os.environ.get("BER_DEVICE"):          # e.g. BER_DEVICE=cpu for local tests
        return os.environ["BER_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def record_texts(df: pd.DataFrame, idx=None) -> list[str]:
    names = (df["name_raw"] if idx is None else df["name_raw"].take(idx)).tolist()
    addrs = (df["addr_raw"] if idx is None else df["addr_raw"].take(idx)).tolist()
    return [f"{n} | {a}" if a else n for n, a in zip(names, addrs)]


def _autocast(dev):
    return torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else torch.autocast("cpu", enabled=False)


# ------------------------------------------------------------------ bi-encoder
def _features(model, texts, dev):
    """Model inputs for a batch (sentence-transformers >= 6 'preprocess'; tensors moved to the device)."""
    prep = model.preprocess(texts) if hasattr(model, "preprocess") else model.tokenize(texts)
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in prep.items()}


def train_bi_encoder(cfg: NeuralConfig, a_texts: list[str], b_texts: list[str], log=print):
    """Multiple-negatives ranking loss (in-batch negatives), one pass over the pairs."""
    from sentence_transformers import SentenceTransformer
    dev = device()
    model = SentenceTransformer(cfg.base_model, device=dev)
    model.max_seq_length = cfg.bi_max_len
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.bi_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    order = np.random.default_rng(0).permutation(len(a_texts))
    steps = len(order) // cfg.bi_batch
    t0 = time.time()
    for step in range(steps):
        idx = order[step * cfg.bi_batch:(step + 1) * cfg.bi_batch]
        fa = _features(model, [a_texts[i] for i in idx], dev)
        fb = _features(model, [b_texts[i] for i in idx], dev)
        with _autocast(dev):
            ea = torch.nn.functional.normalize(model(fa)["sentence_embedding"], dim=-1)
            eb = torch.nn.functional.normalize(model(fb)["sentence_embedding"], dim=-1)
            logits = 20.0 * ea @ eb.T
            labels = torch.arange(len(idx), device=dev)
            loss = (torch.nn.functional.cross_entropy(logits, labels) +
                    torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 200 == 0 or step == steps - 1:
            log(f"  bi-encoder step {step + 1}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    if dev == "cuda":
        model.half()
    return model


@torch.no_grad()
def embed(model, texts: list[str], batch: int, chunk: int = 500_000) -> torch.Tensor:
    """Normalised embeddings written into one preallocated tensor, chunk by chunk (encoding
    everything at once would hold a second full-size copy while concatenating)."""
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    out = torch.empty((len(texts), dim_fn()), dtype=dtype, device=dev)
    for start in range(0, len(texts), chunk):
        e = model.encode(texts[start:start + chunk], batch_size=batch, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
        out[start:start + len(e)] = e.to(dtype)
        del e
    return out


@torch.no_grad()
def embed_records(model, df: pd.DataFrame, idx: np.ndarray, batch: int, chunk: int = 500_000) -> torch.Tensor:
    """Embeddings of df rows `idx`; texts are built per chunk, never for all rows at once."""
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    out = torch.empty((len(idx), dim_fn()), dtype=dtype, device=dev)
    for start in range(0, len(idx), chunk):
        texts = record_texts(df, idx[start:start + chunk])
        e = model.encode(texts, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False)
        out[start:start + len(e)] = e.to(dtype)
        del e, texts
    return out


def load_bi_encoder(path):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(path), device=device())
    model.eval()
    if device() == "cuda":
        model.half()
    return model


@torch.no_grad()
def add_dense(model, cfg: NeuralConfig, s1: pd.DataFrame, r: pd.DataFrame, cand: pd.DataFrame, log=print) -> pd.DataFrame:
    """Union key candidates with dense top-k per country; add dense_cos and dense_rank to every pair."""
    parts = []
    s1_groups = s1.groupby("country").indices
    r_groups = r.groupby("country").indices
    cand = cand.reset_index(drop=True)
    cand_country = r["country"].take(cand["r_idx"].to_numpy()).to_numpy()
    for country, r_idx in r_groups.items():
        t0 = time.time()
        s1_idx = s1_groups.get(country)
        mine = cand[cand_country == country]
        if s1_idx is None:
            continue
        e1 = embed_records(model, s1, s1_idx, cfg.encode_batch)
        er = embed_records(model, r, r_idx, cfg.encode_batch)
        top_s, top_i = [], []
        for start in range(0, len(r_idx), cfg.query_chunk):
            sims = er[start:start + cfg.query_chunk] @ e1.T
            s, i = torch.topk(sims, min(cfg.dense_k, e1.shape[0]), dim=1)
            del sims
            top_s.append(s.cpu()); top_i.append(i.cpu())
        top_i = torch.cat(top_i).numpy()
        k = top_i.shape[1]
        dense = pd.DataFrame({"r_idx": np.repeat(r_idx, k).astype(np.int32), "s1_idx": s1_idx[top_i.ravel()].astype(np.int32),
                              "dense_rank": np.tile(np.arange(1, k + 1, dtype=np.float32), len(r_idx))})
        merged = mine.merge(dense, on=["r_idx", "s1_idx"], how="outer")
        new = merged["block_score"].isna()
        merged["block_score"] = merged["block_score"].fillna(0).astype(np.float32)
        merged["fallback"] = merged["fallback"].fillna(False).astype(bool)
        merged["dense_only"] = new.to_numpy()
        merged["dense_rank"] = merged["dense_rank"].fillna(k + 1).astype(np.float32)
        # cosine for every candidate pair of this country
        pos_r = pd.Series(np.arange(len(r_idx)), index=r_idx)
        pos_1 = pd.Series(np.arange(len(s1_idx)), index=s1_idx)
        pr = torch.from_numpy(pos_r.loc[merged["r_idx"].to_numpy()].to_numpy().copy()).to(er.device)
        p1 = torch.from_numpy(pos_1.loc[merged["s1_idx"].to_numpy()].to_numpy().copy()).to(er.device)
        cos = torch.empty(len(merged), dtype=torch.float32)
        for start in range(0, len(merged), 1_000_000):
            sl = slice(start, start + 1_000_000)
            cos[sl] = (er[pr[sl]].float() * e1[p1[sl]].float()).sum(1).cpu()
        merged["dense_cos"] = cos.numpy()
        # bound the union: best by cosine, best by key score, at most max_cands per record
        g = merged.groupby("r_idx")
        cos_rank = g["dense_cos"].rank(ascending=False, method="first").to_numpy()
        blk_rank = g["block_score"].rank(ascending=False, method="first").to_numpy()
        merged = merged[(cos_rank <= cfg.keep_by_cos) | (blk_rank <= cfg.keep_by_block)]
        merged = merged.sort_values(["r_idx", "dense_cos"], ascending=[True, False])
        merged = merged[merged.groupby("r_idx").cumcount().to_numpy() < cfg.max_cands]
        merged["r_idx"] = merged["r_idx"].astype(np.int32)
        merged["s1_idx"] = merged["s1_idx"].astype(np.int32)
        parts.append(merged.reset_index(drop=True))
        log(f"  dense {country}: +{int(new.sum()):,} new candidates ({len(merged) / len(r_idx):.2f}/record) "
            f"in {time.time() - t0:.0f}s")
        del e1, er
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ cross-encoder
class CrossEncoder:
    def __init__(self, cfg: NeuralConfig, path=None):
        """New cross-encoder from the base model, or a fine-tuned one saved at `path`."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.cfg = cfg
        self.dev = device()
        src = str(path) if path is not None else cfg.base_model
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1).to(self.dev)
        if path is not None:
            self.model.eval()
            if self.dev == "cuda":
                self.model.half()
        self.parallel = torch.nn.DataParallel(self.model) if torch.cuda.device_count() > 1 else self.model

    def save(self, path):
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))

    def _batch(self, a, b):
        x = self.tok(a, b, truncation=True, max_length=self.cfg.ce_max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.dev) for k, v in x.items()}

    def fit(self, a_texts, b_texts, labels, log=print):
        cfg = self.cfg
        n = len(a_texts)
        order = np.random.default_rng(1).permutation(n)
        steps = n // cfg.ce_batch
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.ce_lr, weight_decay=0.01)
        warm = max(1, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(steps, 1)))
        scaler = torch.amp.GradScaler("cuda", enabled=self.dev == "cuda")
        y_all = torch.as_tensor(np.asarray(labels, dtype=np.float32))
        self.parallel.train()
        t0 = time.time()
        for step in range(steps):
            idx = order[step * cfg.ce_batch:(step + 1) * cfg.ce_batch]
            x = self._batch([a_texts[i] for i in idx], [b_texts[i] for i in idx])
            y = y_all[idx].to(self.dev)
            with _autocast(self.dev):
                logits = self.parallel(**x).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 500 == 0 or step == steps - 1:
                log(f"  cross-encoder step {step + 1}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        self.parallel.eval()
        if self.dev == "cuda":
            self.model.half()

    @torch.no_grad()
    def score(self, a_texts, b_texts, log=print) -> np.ndarray:
        out = np.empty(len(a_texts), dtype=np.float32)
        bs = self.cfg.score_batch
        t0 = time.time()
        # sort by length so padded batches are tight
        order = np.argsort([len(a) + len(b) for a, b in zip(a_texts, b_texts)], kind="stable")
        for n, start in enumerate(range(0, len(order), bs)):
            idx = order[start:start + bs]
            x = self._batch([a_texts[i] for i in idx], [b_texts[i] for i in idx])
            with _autocast(self.dev):
                out[idx] = torch.sigmoid(self.parallel(**x).logits.squeeze(-1).float()).cpu().numpy()
            if n % 2000 == 0:
                log(f"  cross-encoder scored {min(start + bs, len(order)):,}/{len(order):,} ({time.time() - t0:.0f}s)")
        return out


def pair_texts(s1: pd.DataFrame, r: pd.DataFrame, cand: pd.DataFrame):
    return (record_texts(s1, cand["s1_idx"].to_numpy()), record_texts(r, cand["r_idx"].to_numpy()))


def score_in_chunks(ce: CrossEncoder, s1, r, cand, chunk=2_000_000, log=print) -> np.ndarray:
    """Cross-encoder probability for each record's top `ce_top` candidates by cosine; NaN elsewhere."""
    key = "dense_cos" if "dense_cos" in cand else "block_score"
    rank = cand.groupby("r_idx")[key].rank(ascending=False, method="first").to_numpy()
    todo = np.flatnonzero(rank <= ce.cfg.ce_top)
    out = np.full(len(cand), np.nan, dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(todo), chunk):
        idx = todo[start:start + chunk]
        a, b = pair_texts(s1, r, cand.iloc[idx])
        out[idx] = ce.score(a, b, log=lambda m: None)
        log(f"  cross-encoder: {min(start + chunk, len(todo)):,}/{len(todo):,} pairs scored "
            f"({len(todo) / max(len(cand), 1):.0%} of candidates, {time.time() - t0:.0f}s)")
    return out
