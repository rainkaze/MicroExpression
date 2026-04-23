from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.datasets.casme3 import CASME3FlowDataset, CASME3_EMOTIONS_7
from src.models.rgbd_motion_attention import RGBDMotionAttentionNet
from src.utils.config import config_get, load_toml_config, parse_config_path
from src.utils.metrics import classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(manifest_path: Path, seed: int, fold: int) -> tuple[list[int], list[int], list[int]]:
    df = pd.read_csv(manifest_path).reset_index(drop=True)
    labels = df["emotion_7"].map({name: index for index, name in enumerate(CASME3_EMOTIONS_7)}).to_numpy()
    groups = df["subject"].astype(str).to_numpy()
    indices = np.arange(len(df))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(indices, labels, groups))
    train_val_idx, test_idx = folds[fold % len(folds)]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 17)
    inner_train_rel, val_rel = next(inner.split(train_val_idx, labels[train_val_idx], groups[train_val_idx]))
    return train_val_idx[inner_train_rel].tolist(), train_val_idx[val_rel].tolist(), test_idx.tolist()


def make_loader(dataset: CASME3FlowDataset, indices: list[int], batch_size: int, train: bool, num_workers: int) -> DataLoader:
    subset = Subset(dataset, indices)
    if not train:
        return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    labels = [dataset.samples[index]["label"] for index in indices]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    for batch in loader:
        motion = batch["flow"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(motion)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
        total_loss += float(loss.item()) * motion.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())
    metrics = classification_metrics(all_labels, all_preds, num_classes=len(CASME3_EMOTIONS_7))
    metrics["loss"] = total_loss / max(1, len(all_labels))
    return metrics


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_confusion_matrix(matrix, classes: list[str], output_path: Path) -> None:
    cm = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CAS(ME)^3 RGB-D Motion Confusion Matrix")
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > threshold else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_training_curves(history: list[dict], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train"]["loss"] for item in history], label="train loss")
    axes[0].plot(epochs, [item["val"]["loss"] for item in history], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [item["train"]["macro_f1"] for item in history], label="train macro-F1")
    axes[1].plot(epochs, [item["val"]["macro_f1"] for item in history], label="val macro-F1")
    axes[1].plot(epochs, [item["val"]["uar"] for item in history], label="val UAR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    config_path, remaining_argv = parse_config_path()
    config = load_toml_config(config_path)
    parser = argparse.ArgumentParser(description="Train CAS(ME)^3 RGB-D motion attention model.")
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--manifest", type=Path, default=config_get(config, "manifest", PROJECT_ROOT / "processed" / "casme3_flow" / "casme3_flow_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=config_get(config, "output_dir", PROJECT_ROOT / "runs" / "casme3_rgbd_motion"))
    parser.add_argument("--epochs", type=int, default=config_get(config, "epochs", 40))
    parser.add_argument("--batch-size", type=int, default=config_get(config, "batch_size", 32))
    parser.add_argument("--lr", type=float, default=config_get(config, "lr", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=config_get(config, "weight_decay", 1e-3))
    parser.add_argument("--fold", type=int, default=config_get(config, "fold", 0))
    parser.add_argument("--seed", type=int, default=config_get(config, "seed", 42))
    parser.add_argument("--num-workers", type=int, default=config_get(config, "num_workers", 0))
    parser.add_argument("--device", default=config_get(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args(remaining_argv)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = CASME3FlowDataset(args.manifest, PROJECT_ROOT, augment=True, path_column="rgbd_flow_path")
    eval_dataset = CASME3FlowDataset(args.manifest, PROJECT_ROOT, augment=False, path_column="rgbd_flow_path")
    train_idx, val_idx, test_idx = split_indices(args.manifest, args.seed, args.fold)
    train_loader = make_loader(train_dataset, train_idx, args.batch_size, True, args.num_workers)
    val_loader = make_loader(eval_dataset, val_idx, args.batch_size, False, args.num_workers)
    test_loader = make_loader(eval_dataset, test_idx, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)
    model = RGBDMotionAttentionNet(num_classes=len(CASME3_EMOTIONS_7)).to(device)
    train_labels = [train_dataset.samples[index]["label"] for index in train_idx]
    counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (len(CASME3_EMOTIONS_7) * max(1, counts[index])) for index in range(len(CASME3_EMOTIONS_7))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    print(f"Samples train/val/test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"Device: {device}")
    print("Train class counts:", {CASME3_EMOTIONS_7[key]: value for key, value in sorted(counts.items())})
    best_val = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train": {k: serializable(v) for k, v in train_metrics.items()}, "val": {k: serializable(v) for k, v in val_metrics.items()}})
        if float(val_metrics["macro_f1"]) > best_val:
            best_val = float(val_metrics["macro_f1"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} mf1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} mf1 {val_metrics['macro_f1']:.4f} uar {val_metrics['uar']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, criterion, device)
    result = {
        "classes": CASME3_EMOTIONS_7,
        "args": {key: serializable(value) for key, value in vars(args).items()},
        "history": history,
        "test": {key: serializable(value) for key, value in test_metrics.items()},
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "best_model.pt")
    save_confusion_matrix(test_metrics["confusion_matrix"], CASME3_EMOTIONS_7, args.output_dir / "confusion_matrix.png")
    save_training_curves(history, args.output_dir / "training_curves.png")
    print(f"Test | loss {test_metrics['loss']:.4f} acc {test_metrics['accuracy']:.4f} macro_f1 {test_metrics['macro_f1']:.4f} uar {test_metrics['uar']:.4f}")
    print(f"Saved: {args.output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
