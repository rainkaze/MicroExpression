from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_per_class": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def summarize_folds(results: list[dict], metric_keys: list[str]) -> dict:
    summary = {}
    for key in metric_keys:
        values = [float(item["test"][key]) for item in results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return summary
