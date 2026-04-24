from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def build_loss(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss_name: {loss_name}")
