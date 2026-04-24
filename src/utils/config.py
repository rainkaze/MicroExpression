from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any


def load_toml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as file:
        return tomllib.load(file)


def parse_config_path() -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    return args.config, remaining


def config_get(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)
