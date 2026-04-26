from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_recognition import run_from_config


if __name__ == "__main__":
    run_from_config(Path("configs/train/casme3/baseline/depth7_baseline_5fold.toml"))
