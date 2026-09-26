"""Reading the challenge TSVs and writing the two submission files.

BER_SUBSAMPLE=<fraction> (tests only) keeps a consistent slice: a hash-selected share of
S1 entities, all train S2/S3 records they own, and the same share of the remaining records.
"""

import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

SUBSAMPLE = float(os.environ.get("BER_SUBSAMPLE", "0") or 0)


def _hashed(ids, frac: float) -> np.ndarray:
    return np.fromiter((zlib.crc32(i.encode()) % 10_000 < frac * 10_000 for i in ids), bool, len(ids))


SOURCE_COLUMNS = ["entity_id", "name", "address", "country"]


def read_tsv(path) -> pd.DataFrame:
    # keep_default_na=False: a business literally named "NA" must stay a string; only empty cells become null
    return pd.read_csv(path, sep="\t", engine="pyarrow", dtype_backend="pyarrow",
                       keep_default_na=False, na_values=[""])


def load_split(split: str, data_dir=DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (s1, r): Source 1 records, and Source 2 + Source 3 records with a `source` column."""
    frames = []
    for k in (1, 2, 3):
        df = read_tsv(Path(data_dir) / split / f"{split}_source{k}.tsv")
        df.columns = SOURCE_COLUMNS
        df["source"] = f"S{k}"
        frames.append(df)
    s1, r = frames[0], pd.concat(frames[1:], ignore_index=True)
    if SUBSAMPLE:
        keep_s1 = _hashed(s1["entity_id"].astype(str).tolist(), SUBSAMPLE)
        r_ids = r["entity_id"].astype(str)
        keep_r = _hashed(r_ids.tolist(), SUBSAMPLE)
        if split == "train":
            pairs = load_pairs(data_dir, subsample=False)
            owner = pairs.set_index("r")["s1"]
            owned = r_ids.map(owner)
            kept_s1 = set(s1["entity_id"].astype(str)[keep_s1])
            keep_r = np.where(owned.notna(), owned.isin(kept_s1), keep_r)
        s1, r = s1[keep_s1].reset_index(drop=True), r[keep_r].reset_index(drop=True)
    return s1, r


def load_pairs(data_dir=DATA_DIR, subsample: bool = True) -> pd.DataFrame:
    """Training ground truth as one row per true (s1, r) pair."""
    gt = read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    gt.columns = ["s1", "matched"]
    gt["matched"] = gt["matched"].fillna("")
    pairs = gt.assign(r=gt["matched"].str.split(",")).explode("r")[["s1", "r"]]
    pairs = pairs[pairs["r"].notna() & (pairs["r"] != "")].astype({"s1": str, "r": str})
    if SUBSAMPLE and subsample:
        pairs = pairs[_hashed(pairs["s1"].tolist(), SUBSAMPLE)]
    return pairs.reset_index(drop=True)


def write_id_lists(path, list_column: str, s1_ids, r_ids, s1_idx: np.ndarray, r_idx: np.ndarray) -> None:
    """One row per S1 id (in `s1_ids` order) with its comma-separated, de-duplicated S2/S3 ids.

    Pairs are given as integer positions into `s1_ids` / `r_ids`, so no per-pair id strings
    are ever materialised (tens of millions of candidate rows).
    """
    s1_idx, r_idx = np.asarray(s1_idx, np.int64), np.asarray(r_idx, np.int64)
    order = np.lexsort((r_idx, s1_idx))
    a, b = s1_idx[order], r_idx[order]
    keep = np.ones(len(a), bool)
    keep[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a, b = a[keep], b[keep]
    bounds = np.searchsorted(a, np.arange(len(s1_ids) + 1))
    r_arr = np.asarray(r_ids, dtype=object)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{list_column}\n")
        for i, sid in enumerate(s1_ids):
            f.write(f"{sid}\t{','.join(r_arr[b[bounds[i]:bounds[i + 1]]])}\n")
