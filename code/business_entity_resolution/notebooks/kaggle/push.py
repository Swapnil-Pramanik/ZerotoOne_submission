"""Push a notebook to Kaggle as a self-contained script kernel.

Usage (from the repository root):
    .venv/bin/python code/business_entity_resolution/notebooks/kaggle/push.py 01_eda

Bundles the *local working copy* of the notebook, src/ and the requirements
files into runner_template.py, and pushes it with the settings in
kaggle/<notebook>/kernel-metadata.json. On Kaggle the runner builds an
isolated environment with the same Python version and pinned packages as the
local .venv, then executes the notebook. Outputs: <notebook>.ipynb and .html.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]  # code/business_entity_resolution
ROOT = PROJECT.parents[1]


def bundle(notebook: str) -> dict:
    files = [
        PROJECT / "notebooks" / f"{notebook}.ipynb",
        PROJECT / "notebooks" / "requirements-dev.txt",
        PROJECT / "requirements.txt",
        *sorted((PROJECT / "src").glob("*.py")),
    ]
    return {str(f.relative_to(ROOT)): f.read_text(encoding="utf-8") for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", help="notebook name without extension, e.g. 01_eda")
    parser.add_argument("--dry-run", action="store_true", help="write the kernel folder and print its path, don't push")
    args = parser.parse_args()

    meta = json.loads((HERE / args.notebook / "kernel-metadata.json").read_text())
    runner = (HERE / "runner_template.py").read_text()
    runner = (runner.replace("__BUNDLE__", repr(bundle(args.notebook)))
                    .replace("__NOTEBOOK__", repr(args.notebook))
                    .replace("__PYTHON_VERSION__", repr(platform.python_version())))

    out = Path(tempfile.mkdtemp(prefix=f"kaggle-{args.notebook}-"))
    (out / "run.py").write_text(runner)
    (out / "kernel-metadata.json").write_text(json.dumps({**meta, "code_file": "run.py", "kernel_type": "script"}, indent=2))
    if args.dry_run:
        print(out)
        return
    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    try:
        subprocess.run([kaggle, "kernels", "push", "-p", str(out)], check=True)
    finally:
        shutil.rmtree(out)


if __name__ == "__main__":
    main()
