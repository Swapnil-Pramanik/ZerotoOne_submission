"""Business Entity Resolution — Amazon ML Challenge 2026.

Pipeline entry point: data -> blocking -> matching -> output.
Placeholder: currently only loads the output TSVs.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "output"


def main() -> None:
    matching_results = pd.read_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t")
    candidate_pairs = pd.read_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t")
    print(f"matching_results: {matching_results.shape}")
    print(f"candidate_pairs: {candidate_pairs.shape}")


if __name__ == "__main__":
    main()
