from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "analysis"

PREFERRED_ORDER = [
    "uv4_baseline_5fold",
    "depth4_baseline_5fold",
    "uvd4_concat_5fold",
    "uvd4_attention_5fold",
    "uvd4_attention_focal_sampler_5fold",
    "uvd4_masked_attention_5fold",
    "uvd4_residual_masked_attention_5fold",
    "uv7_baseline_5fold",
    "depth7_baseline_5fold",
    "uvd7_concat_5fold",
    "uvd7_attention_5fold",
    "uvd7_attention_focal_sampler_5fold",
    "uvd7_masked_attention_5fold",
    "uvd7_residual_masked_attention_5fold",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded(value: float) -> float:
    return round(float(value), 6)


def discover_runs() -> list[str]:
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"Runs directory not found: {RUNS_ROOT}")

    available = {path.parent.name for path in RUNS_ROOT.glob("*/summary.json")}
    ordered = [run_name for run_name in PREFERRED_ORDER if run_name in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def collect_run(run_name: str) -> tuple[dict, list[dict], np.ndarray]:
    run_root = RUNS_ROOT / run_name
    summary = read_json(run_root / "summary.json")
    labels = summary["labels"]
    fold_paths = sorted(run_root.glob("fold_*/result.json"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold result files found for run: {run_name}")

    fold_results = [read_json(path) for path in fold_paths]
    recall = np.asarray([item["test"]["recall_per_class"] for item in fold_results], dtype=float)
    f1 = np.asarray([item["test"]["f1_per_class"] for item in fold_results], dtype=float)
    confusion = np.asarray([item["test"]["confusion_matrix"] for item in fold_results], dtype=int).sum(axis=0)

    summary_row = {
        "run_name": run_name,
        "label_mode": summary["label_mode"],
        "input_mode": summary["input_mode"],
        "num_classes": summary["num_classes"],
    }
    for metric in ["accuracy", "macro_f1", "uar", "precision_macro"]:
        values = summary["metrics"][metric]
        summary_row[f"{metric}_mean"] = rounded(values["mean"])
        summary_row[f"{metric}_std"] = rounded(values["std"])
        summary_row[f"{metric}_folds"] = ";".join(f"{float(value):.6f}" for value in values["values"])

    class_rows = []
    for index, label in enumerate(labels):
        class_rows.append(
            {
                "run_name": run_name,
                "label_mode": summary["label_mode"],
                "input_mode": summary["input_mode"],
                "class": label,
                "recall_mean": rounded(recall[:, index].mean()),
                "recall_std": rounded(recall[:, index].std()),
                "f1_mean": rounded(f1[:, index].mean()),
                "f1_std": rounded(f1[:, index].std()),
            }
        )

    return summary_row, class_rows, confusion


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = writable_path(path)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if output_path != path:
        print(f"Target file is locked, wrote fallback: {output_path}")


def writable_path(path: Path) -> Path:
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    output_path = writable_path(path)
    output_path.write_text(text, encoding=encoding)
    if output_path != path:
        print(f"Target file is locked, wrote fallback: {output_path}")


def write_markdown(path: Path, summary_rows: list[dict], class_rows: list[dict]) -> None:
    lines = [
        "# Scene Flow 实验汇总",
        "",
        "该文件由 `scripts/analysis/summarize_scene_flow_runs.py` 生成，自动扫描 `artifacts/runs/*/summary.json`。",
        "",
        "## 总体指标",
        "",
        "| run | classes | input | Acc | Macro-F1 | UAR |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            "| {run_name} | {num_classes} | {input_mode} | {acc:.4f}+/-{acc_std:.4f} | {f1:.4f}+/-{f1_std:.4f} | {uar:.4f}+/-{uar_std:.4f} |".format(
                run_name=row["run_name"],
                num_classes=row["num_classes"],
                input_mode=row["input_mode"],
                acc=row["accuracy_mean"],
                acc_std=row["accuracy_std"],
                f1=row["macro_f1_mean"],
                f1_std=row["macro_f1_std"],
                uar=row["uar_mean"],
                uar_std=row["uar_std"],
            )
        )

    lines.extend(["", "## 类别指标", ""])
    for run_name in [row["run_name"] for row in summary_rows]:
        lines.extend([f"### {run_name}", "", "| class | Recall | F1 |", "|---|---:|---:|"])
        for row in [item for item in class_rows if item["run_name"] == run_name]:
            lines.append(
                f"| {row['class']} | {row['recall_mean']:.4f}+/-{row['recall_std']:.4f} | {row['f1_mean']:.4f}+/-{row['f1_std']:.4f} |"
            )
        lines.append("")

    write_text(path, "\n".join(lines), encoding="utf-8")


def main() -> None:
    run_names = discover_runs()
    summary_rows = []
    class_rows = []
    confusion_data = {}

    for run_name in run_names:
        summary_row, run_class_rows, confusion = collect_run(run_name)
        summary_rows.append(summary_row)
        class_rows.extend(run_class_rows)
        confusion_data[run_name] = confusion.tolist()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "scene_flow_run_summary.csv", summary_rows)
    write_csv(OUTPUT_ROOT / "scene_flow_run_class_metrics.csv", class_rows)
    write_text(
        OUTPUT_ROOT / "scene_flow_run_confusion_matrices.json",
        json.dumps(confusion_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(OUTPUT_ROOT / "scene_flow_run_summary.md", summary_rows, class_rows)
    print(f"Found {len(run_names)} runs:")
    for run_name in run_names:
        print(f"  - {run_name}")
    print(f"Wrote summaries to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
