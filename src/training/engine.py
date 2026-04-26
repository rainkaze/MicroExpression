from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.datasets import CASME3RecognitionDataset, LABEL_MODES
from src.models import build_model
from src.training.losses import build_loss
from src.training.metrics import classification_metrics, summarize_folds
from src.training.reporting import (
    save_confusion_matrix_plot,
    save_fold_report,
    save_history_csv,
    save_history_plot,
    save_run_report,
    save_summary_json,
)
from src.training.splits import build_subject_aware_splits
from src.utils.runtime import ensure_dir, seed_everything


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return value


def _make_loader(
    dataset: CASME3RecognitionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    progress_desc: str | None = None,
) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, leave=False)

    for batch in iterator:
        inputs = batch["input"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        total_loss += float(loss.item()) * inputs.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())

        if progress_desc is not None:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    metrics = classification_metrics(y_true, y_pred, num_classes=logits.size(1))
    metrics["loss"] = total_loss / max(1, len(y_true))
    return metrics


def _print_epoch_log(run_name: str, fold: int, epoch: int, epochs: int, train_metrics: dict, val_metrics: dict) -> None:
    print(
        f"[{run_name}] fold={fold} epoch={epoch}/{epochs} "
        f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
        f"val_acc={val_metrics['accuracy']:.4f} val_uar={val_metrics['uar']:.4f}",
        flush=True,
    )


def _build_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_weighted_sampler(labels: list[int], num_classes: int, power: float = 1.0) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0
    class_weights = (1.0 / counts) ** power
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_experiment(config: dict[str, Any], project_root: Path) -> Path:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    manifest_path = project_root / data_cfg["manifest_path"]
    input_mode = data_cfg["input_mode"]
    label_mode = data_cfg["label_mode"]
    label_column = "emotion_4" if label_mode == "4class" else "emotion_7"
    labels = LABEL_MODES[label_mode]
    num_classes = len(labels)

    seed_everything(int(train_cfg["seed"]))
    run_name = output_cfg["run_name"]
    run_root = ensure_dir(project_root / output_cfg["root_dir"] / run_name)

    df = pd.read_csv(manifest_path)
    if data_cfg.get("clean_only", True):
        issue_column = "recognition_issues" if "recognition_issues" in df.columns else "issues"
        if issue_column in df.columns:
            df = df[df[issue_column].fillna("").astype(str) == ""].copy()
    df = df[df[label_column].astype(str).str.strip().str.lower().isin(labels)].copy()
    df = df.reset_index(drop=True)

    folds = list(range(5)) if train_cfg.get("run_all_folds", False) else [int(train_cfg["fold"])]
    device_name = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    results: list[dict[str, Any]] = []
    for fold in folds:
        train_idx, val_idx, test_idx = build_subject_aware_splits(df, label_column=label_column, seed=int(train_cfg["seed"]), fold_index=fold)
        fold_dir = ensure_dir(run_root / f"fold_{fold}")

        train_ds = CASME3RecognitionDataset(manifest_path, project_root, input_mode=input_mode, label_mode=label_mode, augment=True, clean_only=data_cfg.get("clean_only", True), indices=train_idx)
        val_ds = CASME3RecognitionDataset(manifest_path, project_root, input_mode=input_mode, label_mode=label_mode, augment=False, clean_only=data_cfg.get("clean_only", True), indices=val_idx)
        test_ds = CASME3RecognitionDataset(manifest_path, project_root, input_mode=input_mode, label_mode=label_mode, augment=False, clean_only=data_cfg.get("clean_only", True), indices=test_idx)

        train_labels = [record.label_index for record in train_ds.records]
        sampler = None
        if train_cfg.get("balanced_sampler", False):
            sampler = _build_weighted_sampler(
                train_labels,
                num_classes=num_classes,
                power=float(train_cfg.get("sampler_power", 1.0)),
            )

        train_loader = _make_loader(
            train_ds,
            int(train_cfg["batch_size"]),
            shuffle=sampler is None,
            num_workers=int(train_cfg["num_workers"]),
            sampler=sampler,
        )
        val_loader = _make_loader(val_ds, int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg["num_workers"]))
        test_loader = _make_loader(test_ds, int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg["num_workers"]))

        model = build_model(
            model_name=model_cfg["name"],
            num_classes=num_classes,
            input_mode=input_mode,
            base_channels=int(model_cfg.get("base_channels", 32)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        ).to(device)

        class_weights = _build_class_weights(train_labels, num_classes, device)
        class_counts = torch.bincount(torch.as_tensor(train_labels, dtype=torch.long), minlength=num_classes).to(device)
        criterion = build_loss(
            loss_name=train_cfg["loss_name"],
            class_weights=class_weights,
            class_counts=class_counts,
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]), eta_min=float(train_cfg["lr"]) * 0.1)

        best_val = -1.0
        best_state = None
        patience = int(train_cfg.get("early_stop_patience", 8))
        stalled = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            print(f"[{run_name}] fold={fold} epoch={epoch}/{int(train_cfg['epochs'])} started", flush=True)
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                progress_desc=f"{run_name} train e{epoch}",
            )
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                progress_desc=f"{run_name} val e{epoch}",
            )
            scheduler.step()
            _print_epoch_log(run_name, fold, epoch, int(train_cfg["epochs"]), train_metrics, val_metrics)

            history.append(
                {
                    "epoch": epoch,
                    "train": {key: _serializable(value) for key, value in train_metrics.items()},
                    "val": {key: _serializable(value) for key, value in val_metrics.items()},
                }
            )

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

        test_metrics = _run_epoch(model, test_loader, criterion, device)
        torch.save(model.state_dict(), fold_dir / "best_model.pt")
        fold_result = {
            "fold": fold,
            "config": {key: _serializable(value) for key, value in config.items()},
            "device": str(device),
            "label_mode": label_mode,
            "input_mode": input_mode,
            "labels": labels,
            "train_distribution": dict(sorted(Counter(train_labels).items())),
            "history": history,
            "test": {key: _serializable(value) for key, value in test_metrics.items()},
        }
        (fold_dir / "result.json").write_text(json.dumps(fold_result, ensure_ascii=False, indent=2), encoding="utf-8")
        save_history_csv(history, fold_dir / "history.csv")
        save_history_plot(history, fold_dir / "training_curves.png")
        save_confusion_matrix_plot(
            test_metrics["confusion_matrix"],
            labels,
            fold_dir / "confusion_matrix.png",
            title=f"{run_name} fold {fold} test confusion matrix",
        )
        save_fold_report(fold_result, fold_dir / "report.md")
        results.append(fold_result)

    summary = {
        "run_name": run_name,
        "input_mode": input_mode,
        "label_mode": label_mode,
        "num_classes": num_classes,
        "labels": labels,
        "folds": [item["fold"] for item in results],
        "metrics": summarize_folds(results, metric_keys=["accuracy", "macro_f1", "uar", "precision_macro"]),
    }
    save_summary_json(summary, run_root / "summary.json")
    save_run_report(summary, run_root / "summary.md")
    return run_root
