from __future__ import annotations

import torch
import torch.nn as nn

from .common import FeatureGate, TinyEncoder
from .masked_scene_flow import SceneFlowMaskedAttentionNet, SceneFlowResidualMaskedAttentionNet


class SingleStreamNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.encoder = TinyEncoder(in_channels, base_channels=base_channels, use_attention=True)
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class SceneFlowAttentionNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.encoder_u = TinyEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_v = TinyEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_d = TinyEncoder(1, base_channels=base_channels, use_attention=True)

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
        feat_u = self.encoder_u(x[:, 0:1])
        feat_v = self.encoder_v(x[:, 1:2])
        feat_d = self.encoder_d(x[:, 2:3])
        fused = torch.cat([feat_u, feat_v, feat_d], dim=1)
        attended = self.fusion_gate(fused)
        mixed = attended + self.mixer(attended)
        return self.head(torch.cat([fused, mixed], dim=1))


def build_model(model_name: str, num_classes: int, input_mode: str, base_channels: int = 32, dropout: float = 0.25) -> nn.Module:
    if model_name == "uv_baseline":
        if input_mode != "uv":
            raise ValueError("uv_baseline requires input_mode='uv'")
        return SingleStreamNet(in_channels=2, num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "depth_baseline":
        if input_mode != "depth":
            raise ValueError("depth_baseline requires input_mode='depth'")
        return SingleStreamNet(in_channels=1, num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "uvd_concat":
        if input_mode != "uvd":
            raise ValueError("uvd_concat requires input_mode='uvd'")
        return SingleStreamNet(in_channels=3, num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "uvd_attention":
        if input_mode != "uvd":
            raise ValueError("uvd_attention requires input_mode='uvd'")
        return SceneFlowAttentionNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "uvd_masked_attention":
        if input_mode != "uvd":
            raise ValueError("uvd_masked_attention requires input_mode='uvd'")
        return SceneFlowMaskedAttentionNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "uvd_residual_masked_attention":
        if input_mode != "uvd":
            raise ValueError("uvd_residual_masked_attention requires input_mode='uvd'")
        return SceneFlowResidualMaskedAttentionNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
