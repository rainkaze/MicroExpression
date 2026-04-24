from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_experiment
from src.utils import load_toml_config


def run_from_config(config_path: Path) -> Path:
    resolved_config = config_path if config_path.is_absolute() else (PROJECT_ROOT / config_path)
    config = load_toml_config(resolved_config)
    run_root = train_experiment(config, project_root=PROJECT_ROOT)
    print(f"Finished run: {run_root}")
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recognition-only models on CAS(ME)^3.")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config.")
    args = parser.parse_args()

    run_from_config(args.config)


if __name__ == "__main__":
    main()
