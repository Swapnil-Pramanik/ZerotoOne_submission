# Business Entity Resolution

## How to reproduce

data -> blocking -> matching -> output

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

1. `BER_DATA_DIR` / `BER_OUTPUT_DIR` environment variables, if set
   (`BER_DATA_DIR` must contain `train/` and `test/`)
2. on Kaggle: the attached dataset containing `train/train_source1.tsv`
   under `/kaggle/input`, and `/kaggle/working/output`
3. locally: `student_resource/dataset/` and `output/` at the repository root

### 3. Run

```bash
.venv/bin/python code/business_entity_resolution/src/main.py
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### 4. Validate

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Notebooks (analysis only)

`notebooks/` holds exploratory analysis; the pipeline in `src/` does not depend
on it. Notebooks import helpers from `src/` and read data via `src/config.py`.

Notebook outputs are stripped on commit by `nbstripout` (see the repository's
`.gitattributes`), except `01_eda.ipynb`, which is committed with its outputs from the
full-data Kaggle run so the analysis and its verdict can be read without re-running it.

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Dataset investigation: sources, noise patterns, ground truth, blocking feasibility, France shift |
| `02_blocking_analysis.ipynb` | Blocking recall vs. candidate-set size |
| `03_error_analysis.ipynb` | Validation errors and F0.5 threshold tuning |

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
.venv/bin/python code/business_entity_resolution/notebooks/kaggle/push.py 01_eda
.venv/bin/kaggle kernels status swapnilpramanik/ber-01-eda
.venv/bin/kaggle kernels output swapnilpramanik/ber-01-eda -p <dir>
```

`push.py 01_eda --dry-run` writes the kernel folder locally without pushing. To test the
runner locally, run its `run.py` with `BER_RUNNER_WORK`, `BER_RUNNER_OUT` and `BER_DATA_DIR` set.
