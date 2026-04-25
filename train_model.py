from __future__ import annotations

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "training" / "train_model.py"
    run_path(str(script), run_name="__main__")
