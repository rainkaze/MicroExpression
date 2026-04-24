from __future__ import annotations

from pathlib import Path
from typing import Any


def train_experiment(config: dict[str, Any], project_root: Path):
    from .engine import train_experiment as _train_experiment

    return _train_experiment(config, project_root)


__all__ = ["train_experiment"]
