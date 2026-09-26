# Business Entity Resolution

For every Source 1 business, find the Source 2 / Source 3 records that describe the same
real-world business. Pipeline: **data → normalisation + learned aliases → key blocking +
dense retrieval (bi-encoder) → pair features + cross-encoder score → LightGBM → one-owner
assignment + F0.5 threshold → output**.

| Version | What | Offline F0.5 | Leaderboard |
| --- | --- | --- | --- |
| v1 | key blocking + LightGBM (CPU) | 0.975 | 0.965 |
| v2 | + fine-tuned multilingual bi-encoder and cross-encoder (GPU) | 0.981 | — (run ran out of RAM in the test phase) |
| v3 | + France self-training (pseudo-labels, learned région aliases, France-only model) | n/a (no French labels) | pending |
| v3 run | (single process) | 0.982 | — (ran out of RAM in the test phase) |
| v4 | v3, test stage in a fresh process | pending | pending |
| v5 | v3, both stages in own processes + hashed blocking keys (−~10 GB) | pending | pending |
| v6 | + sibling stage 2, set selection, France address fix, int32 candidates, keys generated per chunk | pending (dev slice: stage 1 0.9838 → 0.9860) | pending |
| v7 | + LLM judge (Phi-3.5-mini, MIT) on the hardest pairs of a finished run, France prioritised | pending | pending |

## How to reproduce

### 1. Environment

Python 3.14. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Data

Place the organiser-provided `student_resource/` folder (containing
`dataset/train/` and `dataset/test/`) at the repository root. It is gitignored.

Paths are resolved by `src/config.py`, in this order:

1. `BER_DATA_DIR` / `BER_OUTPUT_DIR` / `BER_ARTIFACT_DIR` environment variables, if set
   (`BER_DATA_DIR` must contain `train/` and `test/`)
2. on Kaggle: the attached dataset containing `train/train_source1.tsv` under
   `/kaggle/input`, and `/kaggle/working/output` + `/kaggle/working/artifacts`
3. locally: `student_resource/dataset/`, `output/` and `artifacts/` at the repository root

### 3. Run

```bash
.venv/bin/python code/business_entity_resolution/src/main.py [--fit-frac F] [--val-frac F] [--seed S] [--no-neural] [--no-stage2]
```

`--no-neural` runs the CPU-only v1 pipeline. The default (v2) needs a CUDA GPU for reasonable
speed; the base model is downloaded from the Hugging Face hub on first use.
Test-only switches: `BER_SUBSAMPLE=0.03` (consistent slice of the data), `BER_DEVICE=cpu`,
`--neural-tiny` (tiny neural settings). On macOS, LightGBM and PyTorch load clashing OpenMP
runtimes, so neural runs crash locally there; they run on Kaggle (Linux).

One command validates on train and then predicts test. Each stage runs in its own Python process
(`pipeline.train_stage`, `pipeline.test_stage`; the caller holds only the config and reports): the
training stage saves every model to `artifacts/` (`bundle.pkl` with config, aliases and threshold; `model.txt`;
`bi_encoder/`; `cross_encoder/`) and the test stage (`pipeline.test_stage`) starts in a fresh Python process
that loads them, so no training-phase memory carries over. The test stage can also be re-run on its own from a
saved bundle.

| Written to | File | Content |
| --- | --- | --- |
| `output/` | `matching_results.tsv` | final matches (the leaderboard file) |
| `output/` | `candidate_pairs.tsv` | the exact candidate set the model scored |
| `artifacts/` (gitignored) | `model.txt` | LightGBM model |
| `artifacts/` | `run_report.json` | config, validation F0.5, blocking recall, test summary, top features |
| `artifacts/` | `threshold_table.tsv` | validation F0.5 / precision / recall per threshold |
| `artifacts/` | `matching_results_stage1.tsv` | test matches from stage 1 alone (no France self-training, no stage 2) |
| `artifacts/` | `matching_results_no_france_st.tsv` | final decision, but France scored without self-training (France-only A/B) |
| `artifacts/` | `bundle.pkl`, `model.txt`, `model_stage2.txt`, `bi_encoder/`, `cross_encoder/` | saved training stage: config, aliases, thresholds, decision method, all models |
| `artifacts/` | `test_scores.parquet`, `test_s1_ids.parquet`, `test_r_ids.parquet` | every test candidate (as positions) with stage-1 (before / after France) and stage-2 probabilities, plus the id tables |
| `artifacts/` | `val_scores.parquet` | validation candidates with label, stage-1 and stage-2 probabilities — re-tune decisions without re-running |

The full data needs ~30 GB RAM; it is run on Kaggle via `notebooks/04_v1_model.ipynb`
(see *Running notebooks on Kaggle*).

### 4. Validate the output format

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Method

### Validation scenario (`pipeline.validate`)

Built from train so that offline F0.5 reflects the test set:

* **Ghost entities:** 19% of S1 entities are removed, so their S2/S3 records become
  look-alike distractors with no owner. This lifts distractors per entity from ~1.2
  (train) to ~2.3, the load estimated for test in the EDA.
* **Entity folds:** 1% validation (`val_frac`), 2% fit (`fit_frac`), the rest unused for fitting.
  Aliases and the bi-encoder learn from non-validation pairs. Records follow their entities:
  records touching a validation entity are held out; records touching a fit entity fit
  LightGBM, so every fit entity has *all* of its records (stage 2's sibling features need
  that); a separate sample of other records trains the cross-encoder, so its score is
  honest when LightGBM sees it. Stage-1 scores on fit rows are 2-fold out-of-fold.
* **Metric:** the official macro F0.5 over validation entities, singletons included
  (`evaluate.macro_f05`). The same fold picks the decision threshold.

### Stages

| Stage | Module | What it does |
| --- | --- | --- |
| Normalise | `text.py`, `records.base_fields` | Transliterate (anyascii), lowercase, join dotted initials (`S.A.R.L.` → `sarl`), strip punctuation; first/last address components; script and domain flags |
| Aliases | `aliases.py`, `records.learn_aliases` | Learned from true pairs only: token synonyms (one-token-difference pairs + union-find), positional transliterations for non-Latin names, S2/S3 → S1 state spellings. French street abbreviations are the only hand-written map |
| Country stats | `records.country_stats` | From each scenario's own S1 table (so France works): filler words (≥0.5% of names), stopwords (≥5%), common address words (≥2%), core-token document frequencies |
| Blocking | `blocking.py` | Per country, IDF-weighted sparse overlap of combination keys (hashed into 2^30 buckets, no Python vocabulary) — name token, name-token pair, name × place word, house number × place word, place-word pair. Keys held by more than min(400, max(50, 0.2% of the country's S1)) entities are dropped; top-10 S1 per record (relative cut 0.3). Records with no key match fall back to character-trigram TF-IDF (trigrams in >500 names ignored). Runs in `forkserver` worker processes (safe alongside PyTorch) on row chunks sized by estimated cost, so memory stays bounded at full scale |
| Features | `features.py` | Fuzzy name scores (ratio, token set/sort, partial, Jaro-Winkler; core and spaceless variants), IDF-weighted token overlap, acronym match, shared house numbers, fuzzy address scores, state agreement (NaN when unknown, e.g. France), and context: the pair's rank and margin within the record's candidates, S1 name-collision size, S1 in-degree. No country feature |
| Bi-encoder (v2) | `neural.train_bi_encoder`, `neural.add_dense` | `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) fine-tuned with in-batch negatives on training-fold pairs, reading raw text (native scripts included). Dense top-5 S1 per record (exact GPU matmul, per country) is unioned with the key candidates, then capped: a record keeps its top 5 by cosine plus top 3 by key score, at most 8. Cosine similarity and dense rank become features |
| Cross-encoder (v2) | `neural.CrossEncoder` | Same base model reading both records jointly, fine-tuned on candidate pairs of a record group disjoint from the LightGBM fit (hard negatives); it scores only each record's top 2 candidates by cosine (NaN elsewhere, same rule in validation and test); its probability and rank become features. Scored on 2x T4 with DataParallel |
| Stage 1 | `matching.train` | LightGBM binary classifier, early-stopped on the validation fold; out-of-fold scores on fit rows; prediction aligns features by name |
| Stage 2 (v6) | `stage2.py` | For pairs with stage-1 probability ≥ 0.01: sibling features (best name / address similarity and shared house number with the records stage 1 assigned to the entity, sibling count per source, siblings' best probability) and how that support ranks across the record's candidates; a second LightGBM re-scores them |
| France (v3) | `france.py` | After test scoring: pseudo-labels on French candidates (best candidate ≥ 0.97 → positive; its other candidates and anything ≤ 0.03 → negative), French state aliases (département / city → S1 région) learned from the positives so the state feature works for France, a France-only LightGBM on the pseudo-labels blended 50/50 with the base model. Writes three test variants that differ only in France (see *Outputs*) |
| LLM judge (v7) | `v7.py`, `llm_judge.py` | Post-processing of a finished run. Hard pairs only: each record's best candidate with an uncertain probability plus close second candidates, closest to the threshold first, 40% of the budget reserved for France (wider band). The LLM sees the business, up to 3 records already matched to it and the candidate; P(Yes) is combined with the base probability by a logistic regression fitted on the base run's validation cache (applied only if validation F0.5 improves), or — without a cache — only very confident flips |
| Decision | `pipeline.decide`, `stage2.select_sets` | Each S2/S3 record goes to its single best-scoring S1 entity; then either a global threshold (stage 1 or stage 2) or per-entity set selection by expected F0.5 (keep the top k records, k possibly 0) — whichever validates best |
| Output | `data_io.write_id_lists` | One row per test S1 entity, comma-separated ids, empty when none |

### Source layout

| File | Responsibility |
| --- | --- |
| `src/config.py` | Data / output / artifact paths (env vars, Kaggle, local) |
| `src/data_io.py` | Read TSVs (`sep="\t"`, "NA" stays a string), ground truth → pairs, write id lists, optional consistent subsample |
| `src/text.py` | Normalisation helpers |
| `src/aliases.py` | Alias learning and the `Aliases` container |
| `src/records.py` | Record fields, country statistics, blocking keys (generated on demand), compact arrow storage with per-table column lists |
| `src/blocking.py` | Candidate generation (sparse top-k, fallback), S1-side candidate context |
| `src/features.py` | Pair features |
| `src/neural.py` | GPU models: bi-encoder (training, embedding, dense retrieval) and cross-encoder (training, scoring) |
| `src/matching.py` | LightGBM training, name-aligned prediction, one-owner assignment, threshold search |
| `src/evaluate.py` | Official macro F0.5 and blocking recall |
| `src/france.py` | France self-training (pseudo-labels, state aliases, France model) |
| `src/stage2.py` | Sibling features, stage-2 matrix, best-record assignment, per-entity set selection |
| `src/llm_judge.py` | LLM judge: prompt with sibling context, next-token P(Yes) vs P(No), one model copy per GPU |
| `src/v7.py` | v7 post-processing of a finished run: hard-pair selection (France share), LLM scoring, calibration, re-decision |
| `src/pipeline.py` | Validation scenario and diagnostics (per-country F0.5, false merges by distractor kind), neural stages (each guarded: a failure is logged and skipped), test prediction in record-aligned chunks, `run` |
| `src/main.py` | Command-line entry point |

## Notebooks

`notebooks/` holds analysis and the Kaggle entry point; the pipeline in `src/` does not
depend on it. Notebooks import from `src/` and read data via `src/config.py`.

Notebook outputs are stripped on commit by `nbstripout` (see the repository's
`.gitattributes`), except `01_eda.ipynb`, which is committed with its outputs from the
full-data Kaggle run so the analysis and its verdict can be read without re-running it.

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Dataset investigation: sources, noise patterns, ground truth, blocking feasibility, France shift, verdict |
| `02_blocking_analysis.ipynb` | Blocking recall vs. candidate-set size |
| `03_error_analysis.ipynb` | Validation errors and F0.5 threshold tuning |
| `00_gpu_probe.ipynb` | Checks CUDA, model download and throughput on Kaggle's GPUs |
| `04_v1_model.ipynb` | v1 full run (CPU): validation report, threshold curve, feature importance, test output |
| `05_v2_model.ipynb` | v2 full run (2x T4 GPU): same report plus the neural stages |
| `05_v2_smoke.ipynb` | v2 on a consistent 3% slice with smaller samples — rehearsal before a full run |
| `06_v3_france.ipynb` | v3 full run: v2 + France self-training, three France variants and cached test scores |
| `07_v4.ipynb` | v4 full run: v3 with the two-process (memory-safe) run and chunked GPU embeddings |
| `08_v5.ipynb` | v5 full run: v3 model with both stages in their own processes and hashed blocking keys |
| `09_v6.ipynb` | v6 full run: sibling stage 2, set selection, France fixes, memory-safe |
| `10_llm_probe.ipynb` | LLM judge probe: throughput on 2x T4 and zero-shot separation of true pairs vs look-alikes (Phi-3.5-mini vs Qwen2.5-1.5B) |
| `11_v7_llm.ipynb` | v7: LLM judge on the hardest pairs of a finished run (attached through `kernel_sources`) |

Local setup:

```bash
.venv/bin/pip install -r code/business_entity_resolution/notebooks/requirements-dev.txt
.venv/bin/jupyter lab code/business_entity_resolution/notebooks/
```

### Running notebooks on Kaggle

`notebooks/kaggle/` runs a notebook on Kaggle in an environment identical to the
local one (same Python version, packages from `requirements.txt` + `requirements-dev.txt`):

| File | Role |
| --- | --- |
| `kaggle/push.py` | Bundles the local notebook, `src/*.py` and requirements into one script kernel and pushes it |
| `kaggle/runner_template.py` | The script that runs on Kaggle: restores the bundle, builds an isolated `uv` venv, executes the notebook cell by cell (progress in the kernel log), writes `<notebook>.ipynb` + `.html` to `/kaggle/working` even if a cell fails |
| `kaggle/<notebook>/kernel-metadata.json` | Kaggle kernel settings: private, CPU/GPU, internet on, private dataset attached |

Needs a logged-in Kaggle CLI (`.venv/bin/kaggle auth login`). From the repository root:

```bash
.venv/bin/python code/business_entity_resolution/notebooks/kaggle/push.py 04_v1_model
.venv/bin/kaggle kernels status swapnilpramanik/ber-04-v1-model
.venv/bin/kaggle kernels output swapnilpramanik/ber-04-v1-model -p <dir>
```

`push.py <notebook> --dry-run` writes the kernel folder locally without pushing. To test the
runner locally, run its `run.py` with `BER_RUNNER_WORK`, `BER_RUNNER_OUT` and `BER_DATA_DIR` set.
