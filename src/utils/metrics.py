from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


CLASS_NAMES = ["positive", "negative", "surprise", "others"]


def confusion_matrix_from_predictions(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_report(y_true: List[int], y_pred: List[int], class_names: List[str] | None = None) -> Dict[str, object]:
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = len(class_names)
    cm = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    per_class = []

    precision_values = []
    recall_values = []
    f1_values = []

    for class_idx, class_name in enumerate(class_names):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[class_idx, :].sum()),
            }
        )

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    report = {
        "accuracy": accuracy,
        "uar": float(np.mean(recall_values)) if recall_values else 0.0,
        "uf1": float(np.mean(f1_values)) if f1_values else 0.0,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    return report
