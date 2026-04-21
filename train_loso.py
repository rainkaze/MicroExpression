import copy
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import CASME2FlowDataset, EMOTION_CLASSES
from src.models.sfamnet import SFAMNetLite
from src.utils.logger import setup_logger
from src.utils.metrics import classification_metrics


CONFIG = {
    "processed_dir": "./data/CASME II/processed",
    "csv_path": "./data/CASME II/CASME2-coding-20140508.xlsx",
    "checkpoints_dir": "./checkpoints/7class",
    "log_dir": "./logs",
    "batch_size": 8,
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "early_stop_patience": 12,
    "num_workers": 0,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val_subjects(train_subjects: list[str], fold_id: int) -> tuple[list[str], list[str]]:
    ordered = sorted(train_subjects)
    val_count = max(3, len(ordered) // 6)
    offset = fold_id % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    val_subjects = sorted(rotated[:val_count])
    train_split_subjects = sorted([sub for sub in ordered if sub not in val_subjects])
    return train_split_subjects, val_subjects


def build_weighted_sampler(dataset: CASME2FlowDataset) -> tuple[WeightedRandomSampler, torch.Tensor]:
    label_counts = Counter(sample["label"] for sample in dataset.samples)
    class_weights = torch.ones(len(EMOTION_CLASSES), dtype=torch.float32)
    for label_idx in range(len(EMOTION_CLASSES)):
        count = label_counts.get(label_idx, 0)
        class_weights[label_idx] = 0.0 if count == 0 else len(dataset) / (len(EMOTION_CLASSES) * count)

    sample_weights = [class_weights[sample["label"]].item() for sample in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_weights


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float | np.ndarray]:
    model.eval()
    losses = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for u, v, labels in data_loader:
            u = u.to(device)
            v = v.to(device)
            labels = labels.to(device)

            outputs = model(u, v)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds, len(EMOTION_CLASSES))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_one_fold(fold_id: int, test_subject: str) -> dict[str, object]:
    all_subjects = [str(i).zfill(2) for i in range(1, 27)]
    train_subjects = [sub for sub in all_subjects if sub != test_subject]
    train_split_subjects, val_subjects = split_train_val_subjects(train_subjects, fold_id)

    logger = setup_logger(CONFIG["log_dir"], f"fold_sub{test_subject}")
    logger.info(
        "Fold %s | train=%s | val=%s | test=%s",
        test_subject,
        train_split_subjects,
        val_subjects,
        [test_subject],
    )

    train_dataset = CASME2FlowDataset(
        CONFIG["processed_dir"],
        CONFIG["csv_path"],
        subjects=train_split_subjects,
        augment=True,
    )
    val_dataset = CASME2FlowDataset(
        CONFIG["processed_dir"],
        CONFIG["csv_path"],
        subjects=val_subjects,
        augment=False,
    )
    test_dataset = CASME2FlowDataset(
        CONFIG["processed_dir"],
        CONFIG["csv_path"],
        subjects=[test_subject],
        augment=False,
    )

    sampler, class_weights = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    model = SFAMNetLite(num_classes=len(EMOTION_CLASSES)).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(CONFIG["device"]),
        label_smoothing=CONFIG["label_smoothing"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_losses = []

        for u, v, labels in train_loader:
            u = u.to(CONFIG["device"])
            v = v.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(u, v)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate_model(model, val_loader, criterion, CONFIG["device"])

        val_score = (
            0.7 * float(val_metrics["accuracy"]) +
            0.3 * float(val_metrics["macro_f1"])
        )

        if val_score > best_metric:
            best_metric = val_score
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_acc=%.2f%% val_uar=%.2f%% val_macro_f1=%.2f%%",
            epoch,
            CONFIG["epochs"],
            train_loss,
            float(val_metrics["loss"]),
            100.0 * float(val_metrics["accuracy"]),
            100.0 * float(val_metrics["uar"]),
            100.0 * float(val_metrics["macro_f1"]),
        )

        if patience_counter >= CONFIG["early_stop_patience"]:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError(f"Fold {test_subject} failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path = os.path.join(CONFIG["checkpoints_dir"], f"best_model_fold_{test_subject}.pth")
    torch.save(best_state, checkpoint_path)

    test_metrics = evaluate_model(model, test_loader, criterion, CONFIG["device"])
    logger.info(
        "Fold %s complete | best_epoch=%d | test_acc=%.2f%% | test_uar=%.2f%% | test_macro_f1=%.2f%%",
        test_subject,
        best_epoch,
        100.0 * float(test_metrics["accuracy"]),
        100.0 * float(test_metrics["uar"]),
        100.0 * float(test_metrics["macro_f1"]),
    )

    return {
        "test_subject": test_subject,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "metrics": test_metrics,
    }


def main() -> None:
    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    set_seed(CONFIG["seed"])

    all_subjects = [str(i).zfill(2) for i in range(1, 27)]
    fold_results = []

    for fold_id, test_subject in enumerate(all_subjects):
        fold_result = train_one_fold(fold_id, test_subject)
        fold_results.append(fold_result)

    mean_acc = np.mean([float(result["metrics"]["accuracy"]) for result in fold_results])
    mean_uar = np.mean([float(result["metrics"]["uar"]) for result in fold_results])
    mean_macro_f1 = np.mean([float(result["metrics"]["macro_f1"]) for result in fold_results])

    print("\n" + "=" * 40)
    print(f"7-class LOSO accuracy: {mean_acc * 100:.2f}%")
    print(f"7-class LOSO UAR: {mean_uar * 100:.2f}%")
    print(f"7-class LOSO Macro-F1: {mean_macro_f1 * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
