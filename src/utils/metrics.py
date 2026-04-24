from __future__ import annotations

import numpy as np


def confusion_matrix_from_predictions(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[truth, pred] += 1
    return matrix


def classification_metrics(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> dict[str, float | np.ndarray]:
    if not y_true:
        raise ValueError("Metrics require at least one sample.")

    cm = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0

    recalls = []
    precisions = []
    f1_scores = []

    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fn = cm[class_idx, :].sum() - tp
        fp = cm[:, class_idx].sum() - tp

        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = float(
            2.0 * precision * recall / (precision + recall)
        ) if (precision + recall) else 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "precision_macro": float(np.mean(precisions)),
        "recall_per_class": np.array(recalls, dtype=np.float32),
        "f1_per_class": np.array(f1_scores, dtype=np.float32),
        "confusion_matrix": cm,
    }
