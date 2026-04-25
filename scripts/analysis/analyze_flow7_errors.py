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
    path = PROJECT_ROOT / value.replace("\\", "/").strip()
    if not path.exists():
        raise FileNotFoundError(f"Could not resolve path: {value}")
    return path


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _tensor_stats(tensor: np.ndarray) -> dict[str, float]:
    u = tensor[0]
    v = tensor[1]
    magnitude = tensor[2]
    uv_abs = np.sqrt(np.square(u) + np.square(v))
    return {
        "mag_mean": float(np.mean(magnitude)),
        "mag_p75": float(np.percentile(magnitude, 75)),
        "mag_p95": float(np.percentile(magnitude, 95)),
        "mag_max": float(np.max(magnitude)),
        "uv_abs_mean": float(np.mean(uv_abs)),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rgb(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_sample_figure(row: dict[str, Any], labels: list[str], output_path: Path) -> None:
    tensor = np.load(_resolve_path(str(row["flow_path"]))).astype(np.float32)
    frame_dir = _resolve_path(str(row["frame_dir"]))
    onset_path = find_frame(frame_dir, int(_safe_float(row["onset"])))
    apex_path = find_frame(frame_dir, int(_safe_float(row["apex"])))
    onset = _read_rgb(onset_path) if onset_path else None
    apex = _read_rgb(apex_path) if apex_path else None

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(
        f"{row['sample_id']} true={row['true_label']} pred={row['pred_label']} "
        f"p_true={row['prob_true']:.3f} p_pred={row['prob_pred']:.3f}",
        fontsize=10,
    )
    for ax, image, title in [(axes[0, 0], onset, "onset"), (axes[0, 1], apex, "apex")]:
        ax.set_title(title)
        ax.axis("off")
        if image is not None:
            ax.imshow(image)

    axes[0, 2].bar(labels, row["probabilities"])
    axes[0, 2].tick_params(axis="x", labelrotation=40, labelsize=7)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("probability")

    for ax, channel, title, cmap in [
        (axes[1, 0], tensor[0], "flow u", "coolwarm"),
        (axes[1, 1], tensor[1], "flow v", "coolwarm"),
        (axes[1, 2], tensor[2], "magnitude", "magma"),
    ]:
        ax.set_title(title)
        ax.axis("off")
        ax.imshow(channel, cmap=cmap)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_summary_plots(output_dir: Path, labels: list[str], cm: np.ndarray, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Flow7 Test Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "flow7_confusion_matrix_counts.png", dpi=180)
    plt.close(fig)

    mag_by_label = [[float(row["mag_mean"]) for row in rows if row["true_label"] == label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(mag_by_label, tick_labels=labels, showmeans=True)
    ax.set_title("Motion Magnitude by True 7-Class Label")
    ax.set_ylabel("mean motion magnitude")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "flow7_mag_mean_by_true_label.png", dpi=180)
    plt.close(fig)

    correct_by_label = []
    wrong_by_label = []
    for label in labels:
        label_rows = [row for row in rows if row["true_label"] == label]
        correct_by_label.append(sum(1 for row in label_rows if row["correct"]))
        wrong_by_label.append(sum(1 for row in label_rows if not row["correct"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, correct_by_label, label="correct")
    ax.bar(x, wrong_by_label, bottom=correct_by_label, label="wrong")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.set_ylabel("sample count")
    ax.set_title("Flow7 Correct vs Wrong by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "flow7_correct_wrong_by_class.png", dpi=180)
    plt.close(fig)


def analyze(run_dir: Path, output_dir: Path, max_figures_per_class: int) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    labels: list[str] = summary["labels"]
    rows: list[dict[str, Any]] = []

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
            label_column="emotion_7",
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
                true_indices = batch["label"].numpy()
                for idx, sample_id in enumerate(batch["sample_id"]):
                    raw = row_by_id[str(sample_id)]
                    tensor = np.load(_resolve_path(str(raw["flow_path"]))).astype(np.float32)
                    probabilities = probs[idx].astype(float).tolist()
                    true_idx = int(true_indices[idx])
                    pred_idx = int(preds[idx])
                    sorted_probs = sorted(probabilities, reverse=True)
                    row = {
                        "fold": int(fold),
                        "sample_id": str(sample_id),
                        "subject": str(raw["subject"]),
                        "video_code": str(raw["video_code"]),
                        "true_label": labels[true_idx],
                        "pred_label": labels[pred_idx],
                        "emotion_4": str(raw["emotion_4"]),
                        "correct": labels[true_idx] == labels[pred_idx],
                        "prob_true": float(probabilities[true_idx]),
                        "prob_pred": float(probabilities[pred_idx]),
                        "top1_margin": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0,
                        "onset": raw["onset"],
                        "apex": raw["apex"],
                        "offset": raw["offset"],
                        "duration_onset_apex": _safe_float(raw["apex"]) - _safe_float(raw["onset"]),
                        "duration_onset_offset": _safe_float(raw["offset"]) - _safe_float(raw["onset"]),
                        "au": str(raw.get("au", "")),
                        "objective_class": str(raw.get("objective_class", "")),
                        "frame_dir": str(raw["frame_dir"]),
                        "flow_path": str(raw["flow_path"]),
                        "probabilities": probabilities,
                    }
                    for label, probability in zip(labels, probabilities, strict=False):
                        row[f"prob_{label}"] = float(probability)
                    row.update(_tensor_stats(tensor))
                    rows.append(row)

    csv_rows = [{k: v for k, v in row.items() if k != "probabilities"} for row in rows]
    _write_csv(output_dir / "flow7_test_predictions.csv", csv_rows, list(csv_rows[0].keys()))

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for row in rows:
        cm[labels.index(row["true_label"]), labels.index(row["pred_label"])] += 1
    _save_summary_plots(output_dir, labels, cm, rows)

    figures_dir = output_dir / "class_error_figures"
    rendered = Counter()
    for row in rows:
        if row["correct"]:
            continue
        true_label = row["true_label"]
        if rendered[true_label] >= max_figures_per_class:
            continue
        rendered[true_label] += 1
        filename = f"{true_label}_{rendered[true_label]:02d}_{row['sample_id']}_pred_{row['pred_label']}.png"
        _save_sample_figure(row, labels, figures_dir / filename)

    lines = [
        "# Flow7 Error Analysis",
        "",
        f"- run: `{run_dir.name}`",
        f"- test samples: `{len(rows)}`",
        "",
        "## Per-Class Summary",
        "",
    ]
    for label in labels:
        label_rows = [row for row in rows if row["true_label"] == label]
        correct = sum(1 for row in label_rows if row["correct"])
        pred_counter = Counter(row["pred_label"] for row in label_rows)
        mag_values = [float(row["mag_mean"]) for row in label_rows]
        zero_motion = sum(1 for row in label_rows if float(row["mag_mean"]) == 0.0)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- support: `{len(label_rows)}`")
        lines.append(f"- correct: `{correct}`")
        lines.append(f"- recall: `{correct / len(label_rows) if label_rows else 0:.4f}`")
        lines.append(f"- zero-motion: `{zero_motion}`")
        lines.append(f"- mag_mean: mean=`{_mean(mag_values):.4f}`, std=`{_std(mag_values):.4f}`")
        lines.append("- predicted as: " + ", ".join(f"`{k}`={v}" for k, v in pred_counter.most_common()))
        lines.append("")

    errors = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                errors.append((int(cm[i, j]), true_label, pred_label))
    lines.extend(["## Top Confusions", ""])
    for count, true_label, pred_label in sorted(errors, reverse=True)[:20]:
        lines.append(f"- `{true_label}` -> `{pred_label}`: `{count}`")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `flow7_test_predictions.csv`: per-sample predictions and motion statistics",
            "- `flow7_confusion_matrix_counts.png`: confusion matrix",
            "- `flow7_mag_mean_by_true_label.png`: motion-strength distribution by class",
            "- `flow7_correct_wrong_by_class.png`: correct/wrong counts by class",
            "- `class_error_figures/`: visual panels for representative wrong predictions",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flow7_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote analysis to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze errors for a flow7 recognition run.")
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs_dev2/flow7_main"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/flow7_main_errors"))
    parser.add_argument("--max-figures-per-class", type=int, default=8)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    analyze(run_dir=run_dir, output_dir=output_dir, max_figures_per_class=args.max_figures_per_class)


if __name__ == "__main__":
    main()
