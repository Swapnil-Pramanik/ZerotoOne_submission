"""Business Entity Resolution — Amazon ML Challenge 2026.

Pipeline entry point: data -> blocking -> matching -> output.
Placeholder: currently loads the dataset and the output TSVs.

Data and output locations are resolved in config.py (local or Kaggle).
"""

from pathlib import Path

import pandas as pd

from config import DATA_DIR, OUTPUT_DIR


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", engine="pyarrow", dtype=str)


def main() -> None:
    for split in ("train", "test"):
        for source in (1, 2, 3):
            df = read_tsv(DATA_DIR / split / f"{split}_source{source}.tsv")
            print(f"{split}_source{source}: {df.shape}")
    matching_results = read_tsv(OUTPUT_DIR / "matching_results.tsv")
    candidate_pairs = read_tsv(OUTPUT_DIR / "candidate_pairs.tsv")
    print(f"matching_results: {matching_results.shape}")
    print(f"candidate_pairs: {candidate_pairs.shape}")


if __name__ == "__main__":
    main()
