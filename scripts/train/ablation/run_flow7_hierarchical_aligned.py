from __future__ import annotations

import csv
import json
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import CASME3RecognitionDataset, LABEL_MODES
from src.models import build_model
from src.training.engine import _run_epoch
from src.training.losses import build_loss
from src.training.metrics import classification_metrics
from src.training.reporting import save_confusion_matrix_plot, save_history_csv, save_history_plot
from src.training.splits import build_subject_aware_splits
from src.utils.runtime import ensure_dir, seed_everything

FLOW7_LABELS = ["disgust", "surprise", "others", "fear", "anger", "sad", "happy"]
NEGATIVE7 = {"disgust", "fear", "anger", "sad"}
FOUR_TO_SEVEN = {
    "positive": "happy",
    "surprise": "surprise",
    "others": "others",
}


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs/train/casme3/ablation/flow7_hierarchical_aligned.toml"
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=np.asarray(labels))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_criterion(train_labels: list[int], train_cfg: dict[str, Any], num_classes: int, device: torch.device) -> torch.nn.Module:
    class_counts = torch.bincount(torch.as_tensor(train_labels, dtype=torch.long), minlength=num_classes).to(device)
    return build_loss(
        loss_name=train_cfg["loss_name"],
        class_weights=_class_weights(train_labels, num_classes, device),
        class_counts=class_counts,
        gamma=float(train_cfg.get("focal_gamma", 2.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )


def _make_loader(dataset: CASME3RecognitionDataset, train_cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
    )


def _print_epoch(run_name: str, epoch: int, epochs: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> None:
    print(
        f"[{run_name}] epoch={epoch}/{epochs} "
        f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
        f"val_acc={val_metrics['accuracy']:.4f} val_uar={val_metrics['uar']:.4f}",
        flush=True,
    )


def _train_component(
    *,
    run_name: str,
    label_mode: str,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    train_idx: list[int],
    val_idx: list[int],
    output_dir: Path,
) -> torch.nn.Module:
    labels = LABEL_MODES[label_mode]
    num_classes = len(labels)
    manifest_path = PROJECT_ROOT / data_cfg["manifest_path"]
    device = _device(str(train_cfg["device"]))

    train_ds = CASME3RecognitionDataset(
        manifest_path,
        PROJECT_ROOT,
        input_mode=data_cfg["input_mode"],
        label_mode=label_mode,
        augment=True,
        clean_only=data_cfg.get("clean_only", True),
        indices=train_idx,
    )
    val_ds = CASME3RecognitionDataset(
        manifest_path,
        PROJECT_ROOT,
        input_mode=data_cfg["input_mode"],
        label_mode=label_mode,
        augment=False,
        clean_only=data_cfg.get("clean_only", True),
        indices=val_idx,
    )
    train_loader = _make_loader(train_ds, train_cfg, shuffle=True)
    val_loader = _make_loader(val_ds, train_cfg, shuffle=False)

    model = build_model(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        input_mode=data_cfg["input_mode"],
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    train_labels = [record.label_index for record in train_ds.records]
    criterion = _build_criterion(train_labels, train_cfg, num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["epochs"]),
        eta_min=float(train_cfg["lr"]) * 0.1,
    )

    best_val = -1.0
    best_state = None
    stalled = 0
    history: list[dict[str, Any]] = []
    patience = int(train_cfg["early_stop_patience"])
    epochs = int(train_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        print(f"[{run_name}] epoch={epoch}/{epochs} started", flush=True)
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer, progress_desc=f"{run_name} train e{epoch}")
        val_metrics = _run_epoch(model, val_loader, criterion, device, progress_desc=f"{run_name} val e{epoch}")
        scheduler.step()
        _print_epoch(run_name, epoch, epochs, train_metrics, val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        score = float(val_metrics["macro_f1"])
        if score > best_val:
            best_val = score
            best_state = deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ensure_dir(output_dir)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_history_csv(history, output_dir / "history.csv")
    save_history_plot(history, output_dir / "training_curves.png")
    return model


def _predict(model: torch.nn.Module, dataset: CASME3RecognitionDataset, labels: list[str], device: torch.device) -> dict[str, dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    predictions: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(model(batch["input"].to(device)), dim=1).cpu().numpy()
            pred_indices = np.argmax(probs, axis=1)
            for idx, sample_id in enumerate(batch["sample_id"]):
                pred_index = int(pred_indices[idx])
                predictions[str(sample_id)] = {
                    "label": labels[pred_index],
                    "prob": float(probs[idx, pred_index]),
                }
    return predictions


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_distribution(path: Path, labels: list[str], rows: list[dict[str, Any]]) -> None:
    true_counts = [sum(1 for row in rows if row["true_label"] == label) for label in labels]
    pred_counts = [sum(1 for row in rows if row["pred_label"] == label) for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, true_counts, width=0.36, label="true")
    ax.bar(x + 0.18, pred_counts, width=0.36, label="predicted")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.set_ylabel("sample count")
    ax.set_title("Aligned Hierarchical Flow7 Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    split_cfg = config["split"]
    output_root = ensure_dir(PROJECT_ROOT / config["output"]["root_dir"] / config["output"]["run_name"])
    seed_everything(int(split_cfg["seed"]))

    manifest_path = PROJECT_ROOT / data_cfg["manifest_path"]
    df = pd.read_csv(manifest_path)
    if data_cfg.get("clean_only", True) and "issues" in df.columns:
        df = df[df["issues"].fillna("").astype(str) == ""].copy()
    df = df.reset_index(drop=True)
    train_idx, val_idx, test_idx = build_subject_aware_splits(
        df,
        label_column="emotion_7",
        seed=int(split_cfg["seed"]),
        fold_index=int(split_cfg["fold"]),
    )

    train_subjects = set(df.iloc[train_idx]["subject"].astype(str))
    val_subjects = set(df.iloc[val_idx]["subject"].astype(str))
    test_subjects = set(df.iloc[test_idx]["subject"].astype(str))
    split_audit = {
        "train_test_subject_overlap": len(train_subjects & test_subjects),
        "val_test_subject_overlap": len(val_subjects & test_subjects),
        "train_subjects": len(train_subjects),
        "val_subjects": len(val_subjects),
        "test_subjects": len(test_subjects),
    }

    coarse_model = _train_component(
        run_name="hierarchical_coarse4",
        label_mode="4class",
        model_cfg=config["coarse_model"],
        train_cfg=config["coarse_train"],
        data_cfg=data_cfg,
        train_idx=train_idx,
        val_idx=val_idx,
        output_dir=output_root / "coarse4_fold_0",
    )
    negative_model = _train_component(
        run_name="hierarchical_negative4",
        label_mode="negative4",
        model_cfg=config["negative_model"],
        train_cfg=config["negative_train"],
        data_cfg=data_cfg,
        train_idx=train_idx,
        val_idx=val_idx,
        output_dir=output_root / "negative4_fold_0",
    )

    device = _device(str(config["coarse_train"]["device"]))
    eval_ds = CASME3RecognitionDataset(manifest_path, PROJECT_ROOT, "flow", "7class", False, True, test_idx)
    coarse_ds = CASME3RecognitionDataset(manifest_path, PROJECT_ROOT, "flow", "4class", False, True, test_idx)
    negative_ds = CASME3RecognitionDataset(manifest_path, PROJECT_ROOT, "flow", "negative4", False, True, test_idx)
    coarse_preds = _predict(coarse_model, coarse_ds, LABEL_MODES["4class"], device)
    negative_preds = _predict(negative_model, negative_ds, LABEL_MODES["negative4"], device)

    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for record in eval_ds.records:
        coarse = coarse_preds[record.sample_id]
        if coarse["label"] == "negative":
            negative = negative_preds.get(record.sample_id)
            pred_label = negative["label"] if negative is not None else "disgust"
            negative_prob = negative["prob"] if negative is not None else 0.0
        else:
            pred_label = FOUR_TO_SEVEN[coarse["label"]]
            negative_prob = 0.0
        y_true.append(FLOW7_LABELS.index(record.label_name))
        y_pred.append(FLOW7_LABELS.index(pred_label))
        rows.append(
            {
                "sample_id": record.sample_id,
                "subject": record.subject,
                "true_label": record.label_name,
                "pred_label": pred_label,
                "correct": record.label_name == pred_label,
                "coarse_pred": coarse["label"],
                "coarse_prob": coarse["prob"],
                "negative_sub_pred": negative_preds.get(record.sample_id, {}).get("label", ""),
                "negative_sub_prob": negative_prob,
            }
        )

    metrics = classification_metrics(y_true, y_pred, num_classes=len(FLOW7_LABELS))
    result = {
        "config": config,
        "labels": FLOW7_LABELS,
        "split_audit": split_audit,
        "test": metrics,
    }
    (output_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_predictions(output_root / "hierarchical_predictions.csv", rows)
    _save_distribution(output_root / "hierarchical_distribution.png", FLOW7_LABELS, rows)
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        FLOW7_LABELS,
        output_root / "hierarchical_confusion_matrix.png",
        title="flow7_hierarchical_aligned fold 0 test confusion matrix",
    )

    lines = [
        "# flow7_hierarchical_aligned",
        "",
        "- split: direct `7class` subject-aware fold 0",
        "- route: 4-class coarse classifier, then negative subclass classifier when coarse=`negative`",
        "- train/test subject overlap: `0` expected",
        "",
        "## Metrics",
        "",
        f"- accuracy: `{metrics['accuracy']:.4f}`",
        f"- macro_f1: `{metrics['macro_f1']:.4f}`",
        f"- uar: `{metrics['uar']:.4f}`",
        f"- precision_macro: `{metrics['precision_macro']:.4f}`",
        "",
        "## Recall Per Class",
        "",
    ]
    for label, value in zip(FLOW7_LABELS, metrics["recall_per_class"], strict=False):
        lines.append(f"- {label}: `{value:.4f}`")
    lines.extend(["", "## F1 Per Class", ""])
    for label, value in zip(FLOW7_LABELS, metrics["f1_per_class"], strict=False):
        lines.append(f"- {label}: `{value:.4f}`")
    lines.extend(["", "## Split Audit", ""])
    for key, value in split_audit.items():
        lines.append(f"- {key}: `{value}`")
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote aligned hierarchical run to: {output_root}")


if __name__ == "__main__":
    main()
