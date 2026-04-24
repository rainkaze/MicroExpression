from __future__ import annotations

import torch
import torch.nn as nn

from .common import ChannelSpatialAttention, FeatureGate, MediumEncoder, TinyEncoder


class MotionMultiBranchNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder_u = TinyEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_v = TinyEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_geom = TinyEncoder(2, base_channels=base_channels, use_attention=True)
        fused_dim = self.encoder_u.out_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_u = self.encoder_u(x[:, 0:1])
        feat_v = self.encoder_v(x[:, 1:2])
        feat_geom = self.encoder_geom(x[:, 2:4])
        base = torch.cat([feat_u, feat_v, feat_geom], dim=1)
        gated = self.gate(base) * base
        return self.head(torch.cat([base, gated], dim=1))


class RGBDMotionFusionNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.motion_encoder = TinyEncoder(4, base_channels=base_channels, use_attention=True)
        self.depth_encoder = TinyEncoder(1, base_channels=max(base_channels // 2, 16), use_attention=True)
        motion_dim = self.motion_encoder.out_dim
        depth_dim = self.depth_encoder.out_dim
        self.motion_gate = nn.Sequential(nn.Linear(depth_dim, motion_dim), nn.Sigmoid())
        self.depth_gate = nn.Sequential(nn.Linear(motion_dim, depth_dim), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(motion_dim + depth_dim + motion_dim + depth_dim, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(320, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion = self.motion_encoder(x[:, :4])
        depth = self.depth_encoder(x[:, 4:5])
        motion_context = self.motion_gate(depth) * motion
        depth_context = self.depth_gate(motion) * depth
        fused = torch.cat([motion, depth, motion_context, depth_context], dim=1)
        return self.head(fused)


class MotionMultiBranchV2Net(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder_u = MediumEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_v = MediumEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_mag = MediumEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_ori = MediumEncoder(1, base_channels=base_channels, use_attention=True)

        branch_dim = self.encoder_u.out_dim
        fused_dim = branch_dim * 4
        self.pre_norm = nn.LayerNorm(fused_dim)
        self.branch_gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.cross_mixer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim * 2),
            nn.Linear(fused_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_u = self.encoder_u(x[:, 0:1])
        feat_v = self.encoder_v(x[:, 1:2])
        feat_mag = self.encoder_mag(x[:, 2:3])
        feat_ori = self.encoder_ori(x[:, 3:4])
        fused = torch.cat([feat_u, feat_v, feat_mag, feat_ori], dim=1)
        fused = self.pre_norm(fused)
        gated = fused * self.branch_gate(fused)
        mixed = gated + self.cross_mixer(gated)
        return self.head(torch.cat([fused, mixed], dim=1))


class MotionMultiBranchV3Net(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder_u = MediumEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_v = MediumEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_mag = MediumEncoder(1, base_channels=base_channels, use_attention=True)
        self.encoder_ori = MediumEncoder(1, base_channels=base_channels, use_attention=True)

        branch_dim = self.encoder_u.out_dim
        fused_dim = branch_dim * 4
        interaction_dim = branch_dim * 3

        self.base_norm = nn.LayerNorm(fused_dim)
        self.base_gate = FeatureGate(fused_dim, dropout=dropout * 0.5)
        self.interaction_gate = FeatureGate(interaction_dim, dropout=dropout * 0.5)
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, interaction_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, interaction_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim + interaction_dim),
            nn.Linear(fused_dim + interaction_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_u = self.encoder_u(x[:, 0:1])
        feat_v = self.encoder_v(x[:, 1:2])
        feat_mag = self.encoder_mag(x[:, 2:3])
        feat_ori = self.encoder_ori(x[:, 3:4])

        base = torch.cat([feat_u, feat_v, feat_mag, feat_ori], dim=1)
        base = self.base_norm(base)
        gated_base = self.base_gate(base)

        uv = feat_u * feat_v
        motion_strength = feat_mag * (feat_u + feat_v) * 0.5
        geometry = feat_mag * feat_ori
        interaction = torch.cat([uv, motion_strength, geometry], dim=1)
        interaction = interaction + self.interaction_mlp(self.interaction_gate(interaction))

        return self.head(torch.cat([gated_base, interaction], dim=1))


class MotionSFAMLiteBranch(nn.Module):
    def __init__(self, base_channels: int = 8, use_attention: bool = True) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ChannelSpatialAttention(c1) if use_attention else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            ChannelSpatialAttention(c2) if use_attention else nn.Identity(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.out_dim = c2 * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotionSFAMLiteNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 8,
        dropout: float = 0.45,
    ) -> None:
        super().__init__()
        self.encoder_u = MotionSFAMLiteBranch(base_channels=base_channels, use_attention=True)
        self.encoder_v = MotionSFAMLiteBranch(base_channels=base_channels, use_attention=True)
        self.encoder_mag = MotionSFAMLiteBranch(base_channels=base_channels, use_attention=True)
        branch_dim = self.encoder_u.out_dim
        fused_dim = branch_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_u = self.encoder_u(x[:, 0:1])
        feat_v = self.encoder_v(x[:, 1:2])
        feat_mag = self.encoder_mag(x[:, 2:3])
        return self.head(torch.cat([feat_u, feat_v, feat_mag], dim=1))


class ApexBaselineNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = TinyEncoder(3, base_channels=base_channels, use_attention=False)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_model(model_name: str, num_classes: int, input_mode: str, base_channels: int = 32, dropout: float = 0.2) -> nn.Module:
    if model_name == "motion_multibranch":
        if input_mode != "flow":
            raise ValueError("motion_multibranch requires input_mode='flow'")
        return MotionMultiBranchNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "rgbd_fusion":
        if input_mode != "rgbd_flow":
            raise ValueError("rgbd_fusion requires input_mode='rgbd_flow'")
        return RGBDMotionFusionNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "motion_multibranch_v2":
        if input_mode != "flow":
            raise ValueError("motion_multibranch_v2 requires input_mode='flow'")
        return MotionMultiBranchV2Net(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "motion_multibranch_v3":
        if input_mode != "flow":
            raise ValueError("motion_multibranch_v3 requires input_mode='flow'")
        return MotionMultiBranchV3Net(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "motion_sfam_lite":
        if input_mode != "flow":
            raise ValueError("motion_sfam_lite requires input_mode='flow'")
        return MotionSFAMLiteNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if model_name == "apex_baseline":
        if input_mode != "apex_rgb":
            raise ValueError("apex_baseline requires input_mode='apex_rgb'")
        return ApexBaselineNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
