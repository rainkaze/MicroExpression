from __future__ import annotations

from pathlib import Path
import tomllib


def load_toml_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        return tomllib.load(handle)
