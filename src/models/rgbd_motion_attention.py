from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, ResidualCBAMBlock


class RGBDMotionAttentionNet(nn.Module):
    """Attention CNN for RGB optical flow plus depth-delta motion maps."""

    def __init__(self, num_classes: int = 7, width: int = 32, dropout: float = 0.35) -> None:
        super().__init__()
        self.rgb_motion = nn.Sequential(
            ConvBNAct(4, width, kernel_size=5, stride=2),
            ResidualCBAMBlock(width, width * 2, stride=2),
            ResidualCBAMBlock(width * 2, width * 2),
            ResidualCBAMBlock(width * 2, width * 4, stride=2),
            ResidualCBAMBlock(width * 4, width * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.depth_motion = nn.Sequential(
            ConvBNAct(1, width // 2, kernel_size=5, stride=2),
            ResidualCBAMBlock(width // 2, width, stride=2),
            ResidualCBAMBlock(width, width * 2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        feature_dim = width * 4 + width * 2
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        rgb_flow = motion[:, :4]
        depth_delta = motion[:, 4:5]
        features = torch.cat([self.rgb_motion(rgb_flow), self.depth_motion(depth_delta)], dim=1)
        return self.fusion(features)
