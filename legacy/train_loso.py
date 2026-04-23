import copy
import os
import random
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import CASME2FlowDataset, COARSE_CLASSES, EMOTION_CLASSES
from src.models.sfamnet import SFAMNetLite
from src.utils.logger import setup_logger
from src.utils.metrics import classification_metrics


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


CONFIG = {
    "processed_dir": os.path.join(PROJECT_ROOT, "processed_v2"),
    "csv_path": os.path.join(PROJECT_ROOT, "../data", "CASME II", "CASME2-coding-20140508.xlsx"),
    "checkpoints_dir": os.path.join(PROJECT_ROOT, "../checkpoints_v2"),
    "log_dir": os.path.join(PROJECT_ROOT, "../logs"),
    "batch_size": 16,
    "pretrain_epochs": 0,
    "finetune_epochs": 50,
    "pretrain_lr": 2e-4,
    "finetune_lr": 3e-4,
    "min_lr": 2e-5,
    "weight_decay": 1e-3,
    "temperature": 0.12,
    "class_balance_beta": 0.999,
    "focal_gamma": 0.0,
    "label_smoothing": 0.04,
    "aux_weight": 0.20,
    "early_stop_patience": 18,
    "num_workers": 0,
    "pin_memory": True,
    "seed": 42,
    "protocol": "loso",
    "test_size": 0.20,
    "val_size": 0.20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.5) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.gamma <= 0:
            return F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                label_smoothing=CONFIG["label_smoothing"],
            )
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = torch.pow(1.0 - target_probs, self.gamma)
        weights = self.class_weights[targets]
        return (-weights * focal * target_log_probs).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.12) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        positive_counts = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(positive_counts, min=1.0)
        valid = positive_counts > 0
        if valid.any():
            return -mean_log_prob_pos[valid].mean()
        return -mean_log_prob_pos.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val_subjects(train_subjects: list[str], fold_id: int) -> tuple[list[str], list[str]]:
    ordered = sorted(train_subjects)
    val_count = max(4, len(ordered) // 5)
    offset = fold_id % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    val_subjects = sorted(rotated[:val_count])
    train_split_subjects = sorted([sub for sub in ordered if sub not in val_subjects])
    return train_split_subjects, val_subjects


def build_class_weights(dataset: CASME2FlowDataset, beta: float, key: str) -> torch.Tensor:
    counts = Counter(sample[key] for sample in dataset.samples)
    num_classes = len(EMOTION_CLASSES) if key == "label" else len(COARSE_CLASSES)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for label_idx in range(num_classes):
        count = counts.get(label_idx, 0)
        if count == 0:
            continue
        effective_num = 1.0 - np.power(beta, count)
        weights[label_idx] = (1.0 - beta) / effective_num
    mask = weights > 0
    weights[mask] = weights[mask] / weights[mask].mean()
    return weights


def build_sampler(dataset: CASME2FlowDataset, class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = [class_weights[sample["label"]].item() for sample in dataset.samples]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=max(len(sample_weights), len(sample_weights) * 2),
        replacement=True,
    )


def recompute_geometry(flow: torch.Tensor) -> torch.Tensor:
    u = flow[:, 0:1]
    v = flow[:, 1:2]
    magnitude = torch.clamp(torch.sqrt(u.square() + v.square() + 1e-8), 0.0, 1.0)
    orientation = torch.atan2(v, u) / np.pi
    return torch.cat([u, v, magnitude, orientation], dim=1)


def augment_flow_batch(flow: torch.Tensor) -> torch.Tensor:
    aug = flow.clone()

    flip_h = torch.rand(aug.size(0), device=aug.device) < 0.5
    scale_mask = torch.rand(aug.size(0), device=aug.device) < 0.25
    noise_mask = torch.rand(aug.size(0), device=aug.device) < 0.15

    if flip_h.any():
        aug[flip_h] = torch.flip(aug[flip_h], dims=[3])
        aug[flip_h, 0:1] = -aug[flip_h, 0:1]

    if scale_mask.any():
        scales = torch.empty(scale_mask.sum(), 1, 1, 1, device=aug.device).uniform_(0.93, 1.07)
        aug[scale_mask, 0:2] = aug[scale_mask, 0:2] * scales

    if noise_mask.any():
        aug[noise_mask, 0:2] = aug[noise_mask, 0:2] + 0.01 * torch.randn_like(aug[noise_mask, 0:2])

    aug = recompute_geometry(aug)
    return aug


def evaluate_model(
    model: SFAMNetLite,
    data_loader: DataLoader,
    criterion_fine: nn.Module,
    device: str,
) -> dict[str, float | np.ndarray]:
    model.eval()
    losses = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for flow, labels, _ in data_loader:
            flow = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(flow)
            loss = criterion_fine(logits, labels)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds, len(EMOTION_CLASSES))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def run_pretrain_stage(model: SFAMNetLite, train_loader: DataLoader, logger) -> None:
    criterion = SupConLoss(temperature=CONFIG["temperature"])
    optimizer = optim.AdamW(
        list(model.stream_u.parameters())
        + list(model.stream_v.parameters())
        + list(model.stream_geom.parameters())
        + list(model.fusion_gate.parameters())
        + list(model.embedding_head.parameters())
        + list(model.projection_head.parameters()),
        lr=CONFIG["pretrain_lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    for epoch in range(1, CONFIG["pretrain_epochs"] + 1):
        model.train()
        losses = []
        for flow, labels, _ in train_loader:
            flow = flow.to(CONFIG["device"], non_blocking=True)
            labels = labels.to(CONFIG["device"], non_blocking=True)

            view1 = augment_flow_batch(flow)
            view2 = augment_flow_batch(flow)
            embed1 = model.project(model.encode(view1))
            embed2 = model.project(model.encode(view2))

            features = torch.cat([embed1, embed2], dim=0)
            contrast_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            loss = criterion(features, contrast_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        logger.info(
            "Pretrain [%d/%d] contrastive_loss=%.4f",
            epoch,
            CONFIG["pretrain_epochs"],
            float(np.mean(losses)) if losses else 0.0,
        )


def run_finetune_stage(
    model: SFAMNetLite,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    fine_weights: torch.Tensor,
    coarse_weights: torch.Tensor,
    test_subject: str,
    logger,
) -> dict[str, object]:
    criterion_fine = ClassBalancedFocalLoss(fine_weights.to(CONFIG["device"]), gamma=CONFIG["focal_gamma"])
    criterion_coarse = nn.CrossEntropyLoss(weight=coarse_weights.to(CONFIG["device"]))

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["finetune_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["finetune_epochs"],
        eta_min=CONFIG["min_lr"],
    )

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CONFIG["finetune_epochs"] + 1):
        model.train()
        epoch_losses = []
        for flow, fine_labels, coarse_labels in train_loader:
            flow = flow.to(CONFIG["device"], non_blocking=True)
            fine_labels = fine_labels.to(CONFIG["device"], non_blocking=True)
            coarse_labels = coarse_labels.to(CONFIG["device"], non_blocking=True)
            flow = augment_flow_batch(flow)

            optimizer.zero_grad()
            fine_logits, coarse_logits = model.forward_multitask(flow)
            loss_fine = criterion_fine(fine_logits, fine_labels)
            loss_coarse = criterion_coarse(coarse_logits, coarse_labels)
            loss = loss_fine + CONFIG["aux_weight"] * loss_coarse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate_model(model, val_loader, criterion_fine, CONFIG["device"])
        val_score = (
            0.10 * float(val_metrics["accuracy"])
            + 0.35 * float(val_metrics["uar"])
            + 0.55 * float(val_metrics["macro_f1"])
        )

        if val_score > best_metric:
            best_metric = val_score
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        logger.info(
            "Finetune [%d/%d] train_loss=%.4f val_loss=%.4f val_acc=%.2f%% val_uar=%.2f%% val_macro_f1=%.2f%% lr=%.6f",
            epoch,
            CONFIG["finetune_epochs"],
            train_loss,
            float(val_metrics["loss"]),
            100.0 * float(val_metrics["accuracy"]),
            100.0 * float(val_metrics["uar"]),
            100.0 * float(val_metrics["macro_f1"]),
            optimizer.param_groups[0]["lr"],
        )

        if patience_counter >= CONFIG["early_stop_patience"]:
            logger.info("Early stopping at finetune epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError(f"Fold {test_subject} failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path = os.path.join(CONFIG["checkpoints_dir"], f"best_model_fold_{test_subject}.pth")
    try:
        torch.save(best_state, checkpoint_path)
    except Exception as exc:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = os.path.join(CONFIG["checkpoints_dir"], f"best_model_fold_{test_subject}_{stamp}.pth")
        try:
            torch.save(best_state, fallback_path)
            checkpoint_path = fallback_path
            logger.warning("Primary checkpoint path failed (%s). Saved to %s", exc, fallback_path)
        except Exception as fallback_exc:
            checkpoint_path = ""
            logger.warning("Checkpoint save skipped: %s; fallback also failed: %s", exc, fallback_exc)

    test_metrics = evaluate_model(model, test_loader, criterion_fine, CONFIG["device"])
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

    train_dataset = CASME2FlowDataset(CONFIG["processed_dir"], CONFIG["csv_path"], subjects=train_split_subjects, augment=False)
    val_dataset = CASME2FlowDataset(CONFIG["processed_dir"], CONFIG["csv_path"], subjects=val_subjects, augment=False)
    test_dataset = CASME2FlowDataset(CONFIG["processed_dir"], CONFIG["csv_path"], subjects=[test_subject], augment=False)

    fine_weights = build_class_weights(train_dataset, CONFIG["class_balance_beta"], key="label")
    coarse_weights = build_class_weights(train_dataset, CONFIG["class_balance_beta"], key="coarse_label")
    sampler = build_sampler(train_dataset, fine_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and CONFIG["device"] == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and CONFIG["device"] == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and CONFIG["device"] == "cuda",
    )

    model = SFAMNetLite(num_classes=len(EMOTION_CLASSES), coarse_classes=len(COARSE_CLASSES)).to(CONFIG["device"])
    run_pretrain_stage(model, train_loader, logger)
    return run_finetune_stage(model, train_loader, val_loader, test_loader, fine_weights, coarse_weights, test_subject, logger)


def stratified_indices(labels: list[int], seed: int, test_size: float, val_size: float) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for indices in by_class.values():
        indices = indices.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * test_size))) if n >= 3 else 0
        remaining = n - n_test
        n_val = max(1, int(round(remaining * val_size))) if remaining >= 3 else 0
        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test : n_test + n_val])
        train_indices.extend(indices[n_test + n_val :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def subset_dataset(dataset: CASME2FlowDataset, indices: list[int]) -> CASME2FlowDataset:
    clone = copy.copy(dataset)
    clone.samples = [dataset.samples[idx] for idx in indices]
    return clone


def train_stratified() -> dict[str, object]:
    logger = setup_logger(CONFIG["log_dir"], "stratified_7class")
    dataset = CASME2FlowDataset(CONFIG["processed_dir"], CONFIG["csv_path"], subjects=None, augment=False)
    labels = [sample["label"] for sample in dataset.samples]
    train_idx, val_idx, test_idx = stratified_indices(
        labels,
        seed=CONFIG["seed"],
        test_size=CONFIG["test_size"],
        val_size=CONFIG["val_size"],
    )
    train_dataset = subset_dataset(dataset, train_idx)
    val_dataset = subset_dataset(dataset, val_idx)
    test_dataset = subset_dataset(dataset, test_idx)

    logger.info(
        "Stratified split | train=%d val=%d test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    logger.info("Train label distribution: %s", Counter(sample["label"] for sample in train_dataset.samples))
    logger.info("Val label distribution: %s", Counter(sample["label"] for sample in val_dataset.samples))
    logger.info("Test label distribution: %s", Counter(sample["label"] for sample in test_dataset.samples))

    fine_weights = build_class_weights(train_dataset, CONFIG["class_balance_beta"], key="label")
    coarse_weights = build_class_weights(train_dataset, CONFIG["class_balance_beta"], key="coarse_label")
    sampler = build_sampler(train_dataset, fine_weights)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and CONFIG["device"] == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and CONFIG["device"] == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and CONFIG["device"] == "cuda",
    )

    model = SFAMNetLite(num_classes=len(EMOTION_CLASSES), coarse_classes=len(COARSE_CLASSES)).to(CONFIG["device"])
    run_pretrain_stage(model, train_loader, logger)
    return run_finetune_stage(model, train_loader, val_loader, test_loader, fine_weights, coarse_weights, "stratified", logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 7-class CASME II micro-expression recognizer.")
    parser.add_argument("--protocol", choices=["loso", "stratified"], default=CONFIG["protocol"])
    parser.add_argument("--pretrain-epochs", type=int, default=CONFIG["pretrain_epochs"])
    parser.add_argument("--finetune-epochs", type=int, default=CONFIG["finetune_epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--processed-dir", default=CONFIG["processed_dir"])
    parser.add_argument("--checkpoints-dir", default=CONFIG["checkpoints_dir"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = parser.parse_args()

    CONFIG["protocol"] = args.protocol
    CONFIG["pretrain_epochs"] = args.pretrain_epochs
    CONFIG["finetune_epochs"] = args.finetune_epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["processed_dir"] = args.processed_dir
    CONFIG["checkpoints_dir"] = args.checkpoints_dir
    CONFIG["seed"] = args.seed

    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    set_seed(CONFIG["seed"])

    if CONFIG["protocol"] == "stratified":
        result = train_stratified()
        metrics = result["metrics"]
        print("\n" + "=" * 40)
        print(f"7-class stratified accuracy: {float(metrics['accuracy']) * 100:.2f}%")
        print(f"7-class stratified UAR: {float(metrics['uar']) * 100:.2f}%")
        print(f"7-class stratified Macro-F1: {float(metrics['macro_f1']) * 100:.2f}%")
        print("=" * 40)
        return

    fold_results = []
    all_subjects = [str(i).zfill(2) for i in range(1, 27)]
    for fold_id, test_subject in enumerate(all_subjects):
        fold_results.append(train_one_fold(fold_id, test_subject))

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
