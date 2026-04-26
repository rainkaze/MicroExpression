from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvNormAct, FeatureGate, ResidualStage


class SpatialMaskGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        hidden = max(base_channels // 2, 8)
        self.net = nn.Sequential(
            ConvNormAct(in_channels, hidden, kernel_size=5, stride=2),
            ConvNormAct(hidden, hidden, kernel_size=3, stride=2),
            ConvNormAct(hidden, hidden, kernel_size=3, stride=2),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedBranchEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, mask_mode: str = "suppress") -> None:
        super().__init__()
        if mask_mode not in {"suppress", "residual"}:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")
        self.mask_mode = mask_mode
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.features = nn.Sequential(
            ConvNormAct(in_channels, c1, kernel_size=5, stride=2),
            ResidualStage(c1, c1, stride=1, use_attention=True),
            ResidualStage(c1, c2, stride=2, use_attention=True),
            ResidualStage(c2, c2, stride=1, use_attention=True),
            ResidualStage(c2, c3, stride=2, use_attention=True),
            ResidualStage(c3, c3, stride=1, use_attention=True),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = c3

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        if self.mask_mode == "residual":
            masked = features * (1.0 + 0.5 * mask)
        else:
            masked = features * mask
        return self.pool(masked)


class SceneFlowMaskedAttentionNet(nn.Module):
    """UVD attention model with an explicit motion-depth spatial mask."""

    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.mask = SpatialMaskGenerator(in_channels=3, base_channels=base_channels)
        self.encoder_u = MaskedBranchEncoder(1, base_channels=base_channels)
        self.encoder_v = MaskedBranchEncoder(1, base_channels=base_channels)
        self.encoder_d = MaskedBranchEncoder(1, base_channels=base_channels)

        branch_dim = self.encoder_u.out_dim
        fused_dim = branch_dim * 3
        self.fusion_gate = FeatureGate(fused_dim, dropout=dropout * 0.5)
        self.mixer = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim * 2),
            nn.Linear(fused_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask(x)
        feat_u = self.encoder_u(x[:, 0:1], mask)
        feat_v = self.encoder_v(x[:, 1:2], mask)
        feat_d = self.encoder_d(x[:, 2:3], mask)
        fused = torch.cat([feat_u, feat_v, feat_d], dim=1)
        attended = self.fusion_gate(fused)
        mixed = attended + self.mixer(attended)
        return self.head(torch.cat([fused, mixed], dim=1))


class SceneFlowResidualMaskedAttentionNet(nn.Module):
    """UVD attention model that uses mask as residual enhancement instead of suppression."""

    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.mask = SpatialMaskGenerator(in_channels=3, base_channels=base_channels)
        self.encoder_u = MaskedBranchEncoder(1, base_channels=base_channels, mask_mode="residual")
        self.encoder_v = MaskedBranchEncoder(1, base_channels=base_channels, mask_mode="residual")
        self.encoder_d = MaskedBranchEncoder(1, base_channels=base_channels, mask_mode="residual")

        branch_dim = self.encoder_u.out_dim
        fused_dim = branch_dim * 3
        self.fusion_gate = FeatureGate(fused_dim, dropout=dropout * 0.5)
        self.mixer = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim * 2),
            nn.Linear(fused_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask(x)
        feat_u = self.encoder_u(x[:, 0:1], mask)
        feat_v = self.encoder_v(x[:, 1:2], mask)
        feat_d = self.encoder_d(x[:, 2:3], mask)
        fused = torch.cat([feat_u, feat_v, feat_d], dim=1)
        attended = self.fusion_gate(fused)
        mixed = attended + self.mixer(attended)
        return self.head(torch.cat([fused, mixed], dim=1))
