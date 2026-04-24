from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_subject_aware_splits(df: pd.DataFrame, label_column: str, seed: int, fold_index: int) -> tuple[list[int], list[int], list[int]]:
    labels = df[label_column].to_numpy()
    groups = df["subject"].astype(str).to_numpy()
    indices = np.arange(len(df))

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(indices, labels, groups))
    train_val_idx, test_idx = outer_splits[fold_index]

    train_val_labels = labels[train_val_idx]
    train_val_groups = groups[train_val_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 17)
    inner_train_rel, val_rel = next(inner.split(train_val_idx, train_val_labels, train_val_groups))

    train_idx = train_val_idx[inner_train_rel].tolist()
    val_idx = train_val_idx[val_rel].tolist()
    test_idx = test_idx.tolist()
    return train_idx, val_idx, test_idx
