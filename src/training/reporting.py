from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_history_csv(history: list[dict], output_path: Path) -> None:
    lines = [
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,train_macro_f1,val_macro_f1,train_uar,val_uar"
    ]
    for item in history:
        train_metrics = item["train"]
        val_metrics = item["val"]
        lines.append(
            ",".join(
                [
                    str(item["epoch"]),
                    f"{train_metrics['loss']:.8f}",
                    f"{val_metrics['loss']:.8f}",
                    f"{train_metrics['accuracy']:.8f}",
                    f"{val_metrics['accuracy']:.8f}",
                    f"{train_metrics['macro_f1']:.8f}",
                    f"{val_metrics['macro_f1']:.8f}",
                    f"{train_metrics['uar']:.8f}",
                    f"{val_metrics['uar']:.8f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_history_plot(history: list[dict], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train"]["loss"] for item in history]
    val_loss = [item["val"]["loss"] for item in history]
    train_f1 = [item["train"]["macro_f1"] for item in history]
    val_f1 = [item["val"]["macro_f1"] for item in history]
    train_acc = [item["train"]["accuracy"] for item in history]
    val_acc = [item["val"]["accuracy"] for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_f1, label="train")
    axes[1].plot(epochs, val_f1, label="val")
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, train_acc, label="train")
    axes[2].plot(epochs, val_acc, label="val")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    arr = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fold_report(result: dict, output_path: Path) -> None:
    test_metrics = result["test"]
    lines = [
        f"# Fold {result['fold']} Report",
        "",
        f"- input_mode: `{result['input_mode']}`",
        f"- label_mode: `{result['label_mode']}`",
        f"- accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- macro_f1: `{test_metrics['macro_f1']:.4f}`",
        f"- uar: `{test_metrics['uar']:.4f}`",
        f"- precision_macro: `{test_metrics['precision_macro']:.4f}`",
        "",
        "## Recall Per Class",
    ]

    for label, recall in zip(result["labels"], test_metrics["recall_per_class"], strict=False):
        lines.append(f"- {label}: `{recall:.4f}`")

    lines.extend(
        [
            "",
            "## F1 Per Class",
        ]
    )
    for label, f1 in zip(result["labels"], test_metrics["f1_per_class"], strict=False):
        lines.append(f"- {label}: `{f1:.4f}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_run_report(summary: dict, output_path: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        f"# {summary['run_name']}",
        "",
        f"- input_mode: `{summary['input_mode']}`",
        f"- label_mode: `{summary['label_mode']}`",
        f"- folds: `{summary['folds']}`",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: mean=`{value['mean']:.4f}`, std=`{value['std']:.4f}`, values=`{value['values']}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_json(summary: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
