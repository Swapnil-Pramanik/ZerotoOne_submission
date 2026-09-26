"""Path configuration shared by the pipeline and the notebooks.

Resolves the dataset and output locations for both local runs and Kaggle:
  1. BER_DATA_DIR / BER_OUTPUT_DIR / BER_ARTIFACT_DIR environment variables, if set
  2. on Kaggle, the attached dataset under /kaggle/input and /kaggle/working/{output,artifacts}
  3. locally, student_resource/dataset, output/ and artifacts/ at the repository root

output/ holds only the two submission TSVs; artifacts/ holds the model, run report
and threshold table.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.exists()


def _find_kaggle_dataset() -> Path:
    for marker in sorted(KAGGLE_INPUT.rglob("train/train_source1.tsv")):
        return marker.parent.parent
    raise FileNotFoundError("no attached Kaggle dataset contains train/train_source1.tsv")


def _data_dir() -> Path:
    if "BER_DATA_DIR" in os.environ:
        return Path(os.environ["BER_DATA_DIR"])
    if ON_KAGGLE:
        return _find_kaggle_dataset()
    return ROOT / "student_resource" / "dataset"


def _output_dir() -> Path:
    if "BER_OUTPUT_DIR" in os.environ:
        return Path(os.environ["BER_OUTPUT_DIR"])
    if ON_KAGGLE:
        return Path("/kaggle/working/output")
    return ROOT / "output"


def _artifact_dir() -> Path:
    if "BER_ARTIFACT_DIR" in os.environ:
        return Path(os.environ["BER_ARTIFACT_DIR"])
    if ON_KAGGLE:
        return Path("/kaggle/working/artifacts")
    return ROOT / "artifacts"


DATA_DIR = _data_dir()
OUTPUT_DIR = _output_dir()
ARTIFACT_DIR = _artifact_dir()
