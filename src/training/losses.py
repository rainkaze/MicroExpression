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


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor, label_smoothing: float = 0.0) -> None:
        super().__init__()
        counts = torch.clamp(class_counts.float(), min=1.0)
        self.register_buffer("log_counts", torch.log(counts))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_counts.to(logits.device)
        return F.cross_entropy(adjusted_logits, target, label_smoothing=self.label_smoothing)


def build_loss(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    class_counts: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, label_smoothing=label_smoothing)
    if loss_name == "balanced_softmax":
        if class_counts is None:
            raise ValueError("balanced_softmax requires class_counts")
        return BalancedSoftmaxLoss(class_counts=class_counts, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss_name: {loss_name}")
