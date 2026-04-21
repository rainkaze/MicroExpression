import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, ResidualCBAMBlock


class FlowEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, 32, kernel_size=5, stride=2)
        self.layer1 = nn.Sequential(
            ResidualCBAMBlock(32, 64, stride=2),
            ResidualCBAMBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResidualCBAMBlock(64, 128, stride=2),
            ResidualCBAMBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResidualCBAMBlock(128, 192, stride=2),
            ResidualCBAMBlock(192, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.proj(x)


class SFAMNetLite(nn.Module):
    """
    Three-stream adaptation inspired by SFAMNet:
    1. horizontal motion (u)
    2. vertical motion (v)
    3. motion geometry (magnitude + orientation)

    We replace the original spotting head with a feasible auxiliary coarse-emotion head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        coarse_classes: int = 4,
        projection_dim: int = 96,
    ) -> None:
        super().__init__()
        self.stream_u = FlowEncoder(in_channels=1, embed_dim=128)
        self.stream_v = FlowEncoder(in_channels=1, embed_dim=128)
        self.stream_geom = FlowEncoder(in_channels=2, embed_dim=128)

        self.fusion_gate = nn.Sequential(
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, 384),
            nn.Sigmoid(),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(768, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Linear(192, projection_dim),
        )
        self.classifier_fine = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )
        self.classifier_coarse = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, coarse_classes),
        )

    def encode(self, flow: torch.Tensor) -> torch.Tensor:
        u = flow[:, 0:1, :, :]
        v = flow[:, 1:2, :, :]
        geom = flow[:, 2:4, :, :]

        feat_u = self.stream_u(u)
        feat_v = self.stream_v(v)
        feat_geom = self.stream_geom(geom)

        base = torch.cat([feat_u, feat_v, feat_geom], dim=1)
        gate = self.fusion_gate(base)
        gated = gate * torch.cat(
            [
                feat_u * feat_v,
                feat_u * feat_geom,
                feat_v * feat_geom,
            ],
            dim=1,
        )
        combined = torch.cat([base, gated], dim=1)
        return self.embedding_head(combined)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(embeddings), dim=1)

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier_fine(embeddings)

    def classify_aux(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier_coarse(embeddings)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode(flow)
        return self.classify(embeddings)

    def forward_multitask(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(flow)
        return self.classify(embeddings), self.classify_aux(embeddings)
