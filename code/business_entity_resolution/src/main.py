"""Business Entity Resolution — Amazon ML Challenge 2026.

Entry point: data -> blocking -> matching -> output.

    python src/main.py                # validate on train (GPU models, stage 2), then write output/*.tsv for test
    python src/main.py --no-neural    # CPU-only pipeline

Paths come from config.py (local or Kaggle). output/ receives matching_results.tsv
and candidate_pairs.tsv; artifacts/ receives model.txt, run_report.json and
threshold_table.tsv.
"""

import argparse
import json

from config import ARTIFACT_DIR, DATA_DIR, OUTPUT_DIR
from pipeline import RunConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-frac", type=float, default=RunConfig.fit_frac,
                        help="share of S1 entities whose records fit LightGBM (default %(default)s)")
    parser.add_argument("--val-frac", type=float, default=RunConfig.val_frac,
                        help="share of S1 entities held out for validation (default %(default)s)")
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--no-neural", action="store_true", help="CPU-only pipeline (no bi-/cross-encoder)")
    parser.add_argument("--no-stage2", action="store_true", help="skip the sibling stage and set selection")
    parser.add_argument("--neural-tiny", action="store_true", help="tiny neural settings for quick local tests")
    args = parser.parse_args()
    print(f"data {DATA_DIR} -> output {OUTPUT_DIR}, artifacts {ARTIFACT_DIR}")
    cfg = RunConfig(fit_frac=args.fit_frac, val_frac=args.val_frac, seed=args.seed, use_neural=not args.no_neural,
                    stage2=not args.no_stage2)
    if args.neural_tiny and cfg.neural is not None:
        cfg.neural.bi_train_pairs, cfg.neural.bi_batch = 2_000, 64
        cfg.neural.ce_train_records, cfg.neural.ce_max_pairs, cfg.neural.ce_batch = 1_000, 4_000, 32
        cfg.neural.encode_batch, cfg.neural.score_batch = 256, 256
    report = run(cfg, OUTPUT_DIR, ARTIFACT_DIR)
    print(json.dumps({k: report[k] for k in ("validation", "method", "blocking", "neural")}, indent=2))


if __name__ == "__main__":
    main()
