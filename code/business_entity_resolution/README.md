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
