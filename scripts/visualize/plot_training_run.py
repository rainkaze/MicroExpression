from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reporting import (
    save_confusion_matrix_plot,
    save_fold_report,
    save_history_csv,
    save_history_plot,
    save_run_report,
)


def render_run(run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    save_run_report(summary, run_dir / "summary.md")
    labels = summary["labels"]

    for fold in summary["folds"]:
        fold_dir = run_dir / f"fold_{fold}"
        result = json.loads((fold_dir / "result.json").read_text(encoding="utf-8"))
        result.setdefault("labels", labels)
        save_history_csv(result["history"], fold_dir / "history.csv")
        save_history_plot(result["history"], fold_dir / "training_curves.png")
        save_confusion_matrix_plot(
            result["test"]["confusion_matrix"],
            labels,
            fold_dir / "confusion_matrix.png",
            title=f"{summary['run_name']} Fold {fold} Test Confusion Matrix",
        )
        save_fold_report(result, fold_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots and reports for a training run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to a run directory containing summary.json.")
    args = parser.parse_args()
    render_run((PROJECT_ROOT / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir)


if __name__ == "__main__":
    main()
