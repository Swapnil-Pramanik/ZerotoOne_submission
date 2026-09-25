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
To read data from elsewhere, set `BER_DATA_DIR` to a folder containing
`train/` and `test/`.

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
