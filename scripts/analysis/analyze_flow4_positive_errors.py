from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import CASME3RecognitionDataset
from src.models import build_model
from src.preprocess.motion import find_frame
from src.training.splits import build_subject_aware_splits


def _resolve_path(value: str) -> Path:
    normalized = value.replace("\\", "/").strip()
    path = PROJECT_ROOT / normalized
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not resolve path: {value}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _tensor_stats(tensor: np.ndarray) -> dict[str, float]:
    u = tensor[0]
    v = tensor[1]
    magnitude = tensor[2]
    orientation = tensor[3]
    uv_abs = np.sqrt(np.square(u) + np.square(v))
    active = magnitude > float(np.percentile(magnitude, 75))
    return {
        "u_abs_mean": float(np.mean(np.abs(u))),
        "v_abs_mean": float(np.mean(np.abs(v))),
        "uv_abs_mean": float(np.mean(uv_abs)),
        "mag_mean": float(np.mean(magnitude)),
        "mag_std": float(np.std(magnitude)),
        "mag_p75": float(np.percentile(magnitude, 75)),
        "mag_p95": float(np.percentile(magnitude, 95)),
        "mag_max": float(np.max(magnitude)),
        "active_ratio_p75": float(np.mean(active)),
        "ori_abs_mean": float(np.mean(np.abs(orientation))),
    }


def _read_rgb(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_sample_figure(record: dict[str, Any], output_path: Path) -> None:
    tensor = np.load(_resolve_path(str(record["flow_path"]))).astype(np.float32)
    frame_dir = _resolve_path(str(record["frame_dir"]))
    onset_path = find_frame(frame_dir, int(_safe_float(record["onset"])))
    apex_path = find_frame(frame_dir, int(_safe_float(record["apex"])))
    onset = _read_rgb(onset_path) if onset_path else None
    apex = _read_rgb(apex_path) if apex_path else None

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle(
        f"{record['sample_id']} true={record['true_label']} pred={record['pred_label']} "
        f"p_true={record['prob_true']:.3f} p_pred={record['prob_pred']:.3f}",
        fontsize=10,
    )

    for ax, image, title in [
        (axes[0, 0], onset, "onset"),
        (axes[0, 1], apex, "apex"),
    ]:
        ax.set_title(title)
        ax.axis("off")
        if image is not None:
            ax.imshow(image)

    axes[0, 2].bar(record["labels"], record["probabilities"])
    axes[0, 2].set_title("probability")
    axes[0, 2].tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0, 2].set_ylim(0, 1)

    channel_specs = [
        (tensor[0], "flow u", "coolwarm"),
        (tensor[1], "flow v", "coolwarm"),
        (tensor[2], "magnitude", "magma"),
    ]
    for ax, (channel, title, cmap) in zip(axes[1], channel_specs, strict=False):
        ax.set_title(title)
        ax.axis("off")
        ax.imshow(channel, cmap=cmap)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_summary_plots(output_dir: Path, labels: list[str], positive_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_counts = Counter(row["pred_label"] for row in positive_rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [pred_counts.get(label, 0) for label in labels], color=["#4c566a", "#88c0d0", "#d08770", "#a3be8c"])
    ax.set_title("True Positive-Class Samples: Predicted Labels")
    ax.set_ylabel("sample count")
    fig.tight_layout()
    fig.savefig(output_dir / "positive_prediction_distribution.png", dpi=180)
    plt.close(fig)

    groups = ["correct", "pred_negative", "pred_surprise", "pred_others"]
    group_rows = [
        [row for row in positive_rows if row["correct"]],
        [row for row in positive_rows if row["pred_label"] == "negative"],
        [row for row in positive_rows if row["pred_label"] == "surprise"],
        [row for row in positive_rows if row["pred_label"] == "others"],
    ]
    for key, ylabel, filename in [
        ("mag_mean", "mean motion magnitude", "positive_mag_mean_boxplot.png"),
        ("mag_p95", "p95 motion magnitude", "positive_mag_p95_boxplot.png"),
        ("duration_onset_apex", "onset-apex duration", "positive_duration_boxplot.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        values = [[float(row[key]) for row in rows] for rows in group_rows]
        ax.boxplot(values, tick_labels=groups, showmeans=True)
        ax.set_title(f"Positive Samples: {key}")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    colors = {"positive": "#88c0d0", "negative": "#4c566a", "surprise": "#d08770", "others": "#a3be8c"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in labels:
        rows = [row for row in positive_rows if row["pred_label"] == label]
        ax.scatter(
            [float(row["mag_mean"]) for row in rows],
            [float(row["prob_positive"]) for row in rows],
            label=label,
            alpha=0.8,
            s=42,
            color=colors.get(label),
        )
    ax.set_title("Positive Samples: Motion Strength vs Positive Probability")
    ax.set_xlabel("mean motion magnitude")
    ax.set_ylabel("predicted probability of positive")
    ax.legend(title="predicted")
    fig.tight_layout()
    fig.savefig(output_dir / "positive_motion_vs_probability.png", dpi=180)
    plt.close(fig)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def analyze(run_dir: Path, output_dir: Path, max_figures: int) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    labels: list[str] = summary["labels"]
    positive_index = labels.index("positive")
    all_rows: list[dict[str, Any]] = []

    for fold in summary["folds"]:
        fold_dir = run_dir / f"fold_{fold}"
        result = json.loads((fold_dir / "result.json").read_text(encoding="utf-8"))
        config = result["config"]
        data_cfg = config["data"]
        model_cfg = config["model"]
        train_cfg = config["train"]
        manifest_path = PROJECT_ROOT / data_cfg["manifest_path"]
        df = pd.read_csv(manifest_path)
        if data_cfg.get("clean_only", True) and "issues" in df.columns:
            df = df[df["issues"].fillna("").astype(str) == ""].copy()
        df = df.reset_index(drop=True)
        _, _, test_idx = build_subject_aware_splits(
            df,
            label_column="emotion_4",
            seed=int(train_cfg["seed"]),
            fold_index=int(fold),
        )

        dataset = CASME3RecognitionDataset(
            manifest_path,
            PROJECT_ROOT,
            input_mode=data_cfg["input_mode"],
            label_mode=data_cfg["label_mode"],
            augment=False,
            clean_only=data_cfg.get("clean_only", True),
            indices=test_idx,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(
            model_name=model_cfg["name"],
            num_classes=len(labels),
            input_mode=data_cfg["input_mode"],
            base_channels=int(model_cfg.get("base_channels", 32)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        ).to(device)
        model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location=device))
        model.eval()

        row_by_id = {record.sample_id: record.raw_row for record in dataset.records}
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["input"].to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                true_labels = batch["label"].numpy()
                for idx, sample_id in enumerate(batch["sample_id"]):
                    raw = row_by_id[str(sample_id)]
                    tensor = np.load(_resolve_path(str(raw["flow_path"]))).astype(np.float32)
                    stats = _tensor_stats(tensor)
                    pred_idx = int(preds[idx])
                    true_idx = int(true_labels[idx])
                    probabilities = probs[idx].astype(float).tolist()
                    sorted_probs = sorted(probabilities, reverse=True)
                    row = {
                        "fold": int(fold),
                        "sample_id": str(sample_id),
                        "subject": str(raw["subject"]),
                        "video_code": str(raw["video_code"]),
                        "emotion_7": str(raw["emotion_7"]),
                        "true_label": labels[true_idx],
                        "pred_label": labels[pred_idx],
                        "correct": labels[true_idx] == labels[pred_idx],
                        "prob_true": float(probabilities[true_idx]),
                        "prob_pred": float(probabilities[pred_idx]),
                        "prob_negative": float(probabilities[labels.index("negative")]),
                        "prob_positive": float(probabilities[labels.index("positive")]),
                        "prob_surprise": float(probabilities[labels.index("surprise")]),
                        "prob_others": float(probabilities[labels.index("others")]),
                        "top1_margin": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0,
                        "onset": raw["onset"],
                        "apex": raw["apex"],
                        "offset": raw["offset"],
                        "duration_onset_offset": _safe_float(raw["offset"]) - _safe_float(raw["onset"]),
                        "duration_onset_apex": _safe_float(raw["apex"]) - _safe_float(raw["onset"]),
                        "frame_count": _safe_float(raw.get("frame_count", 0)),
                        "au": str(raw.get("au", "")),
                        "objective_class": str(raw.get("objective_class", "")),
                        "frame_dir": str(raw["frame_dir"]),
                        "flow_path": str(raw["flow_path"]),
                        "labels": labels,
                        "probabilities": probabilities,
                    }
                    row.update(stats)
                    all_rows.append(row)

    positive_rows = [row for row in all_rows if row["true_label"] == "positive"]
    positive_csv_rows = [{k: v for k, v in row.items() if k not in {"labels", "probabilities"}} for row in positive_rows]
    all_csv_rows = [{k: v for k, v in row.items() if k not in {"labels", "probabilities"}} for row in all_rows]
    fieldnames = list(all_csv_rows[0].keys()) if all_csv_rows else []
    _write_csv(output_dir / "all_test_predictions.csv", all_csv_rows, fieldnames)
    _write_csv(output_dir / "positive_test_predictions.csv", positive_csv_rows, fieldnames)

    figures_dir = output_dir / "positive_figures"
    for index, row in enumerate(positive_rows[:max_figures]):
        suffix = "correct" if row["correct"] else f"pred_{row['pred_label']}"
        _save_sample_figure(row, figures_dir / f"{index:03d}_{row['sample_id']}_{suffix}.png")
    _save_summary_plots(output_dir, labels, positive_rows)

    pred_counter = Counter(row["pred_label"] for row in positive_rows)
    correct_rows = [row for row in positive_rows if row["correct"]]
    wrong_rows = [row for row in positive_rows if not row["correct"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positive_rows:
        grouped[row["pred_label"]].append(row)

    stat_keys = [
        "mag_mean",
        "mag_p75",
        "mag_p95",
        "mag_max",
        "uv_abs_mean",
        "active_ratio_p75",
        "duration_onset_offset",
        "duration_onset_apex",
        "top1_margin",
        "prob_true",
    ]
    lines = [
        "# Positive Error Analysis",
        "",
        f"- run: `{run_dir.name}`",
        f"- total test samples: `{len(all_rows)}`",
        f"- positive test samples: `{len(positive_rows)}`",
        f"- positive correct: `{len(correct_rows)}`",
        f"- positive wrong: `{len(wrong_rows)}`",
        "",
        "## Positive Prediction Distribution",
    ]
    for label in labels:
        lines.append(f"- predicted `{label}`: `{pred_counter.get(label, 0)}`")

    lines.extend(["", "## Positive Motion Statistics"])
    for name, rows in [("correct_positive", correct_rows), ("wrong_positive", wrong_rows)]:
        lines.append(f"### {name}")
        lines.append("")
        for key in stat_keys:
            values = [float(row[key]) for row in rows]
            lines.append(f"- {key}: mean=`{_mean(values):.4f}`, std=`{_std(values):.4f}`")
        lines.append("")

    lines.append("## Grouped By Predicted Class")
    for label in labels:
        rows = grouped.get(label, [])
        lines.append(f"### predicted {label}")
        lines.append("")
        lines.append(f"- samples: `{len(rows)}`")
        for key in stat_keys:
            values = [float(row[key]) for row in rows]
            lines.append(f"- {key}: mean=`{_mean(values):.4f}`, std=`{_std(values):.4f}`")
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `all_test_predictions.csv`: all fold test predictions with motion statistics",
            "- `positive_test_predictions.csv`: only true-positive-class samples",
            "- `positive_figures/`: onset/apex/probability/flow-channel panels for positive samples",
            "- `positive_prediction_distribution.png`: distribution of predicted labels for positive samples",
            "- `positive_mag_mean_boxplot.png`: motion magnitude comparison by predicted label",
            "- `positive_motion_vs_probability.png`: scatter plot of motion strength and positive probability",
            "",
            "## Interpretation Hints",
            "",
            "- If wrong positive samples have lower `mag_mean` or `mag_p95`, the positive class is likely weak-motion limited.",
            "- If wrong positive samples have high `prob_surprise`, the issue is class-boundary similarity between happy and surprise.",
            "- If wrong positive samples have high `prob_negative`, the issue is either weak positive evidence or dominant-class attraction.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "positive_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote analysis to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze positive-class errors for a flow4 recognition run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs_dev2/flow4_main_balanced_softmax_5fold"),
        help="Training run directory containing summary.json and fold checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/flow4_main_balanced_softmax_positive_errors"),
        help="Directory for CSV reports and visualizations.",
    )
    parser.add_argument("--max-figures", type=int, default=80, help="Maximum positive sample figures to render.")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    analyze(run_dir=run_dir, output_dir=output_dir, max_figures=args.max_figures)


if __name__ == "__main__":
    main()
