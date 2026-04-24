from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_attention: bool = False) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = SEBlock(out_channels) if use_attention else nn.Identity()
        self.act = nn.GELU()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return self.act(x + identity)


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, use_attention: bool = False) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.net = nn.Sequential(
            ConvNormAct(in_channels, c1, kernel_size=5, stride=2),
            ResidualStage(c1, c1, stride=1, use_attention=use_attention),
            ResidualStage(c1, c2, stride=2, use_attention=use_attention),
            ResidualStage(c2, c2, stride=1, use_attention=use_attention),
            ResidualStage(c2, c3, stride=2, use_attention=use_attention),
            ResidualStage(c3, c3, stride=1, use_attention=use_attention),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.out_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MediumEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, use_attention: bool = False) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6
        self.net = nn.Sequential(
            ConvNormAct(in_channels, c1, kernel_size=5, stride=2),
            ResidualStage(c1, c1, stride=1, use_attention=use_attention),
            ResidualStage(c1, c2, stride=2, use_attention=use_attention),
            ResidualStage(c2, c2, stride=1, use_attention=use_attention),
            ResidualStage(c2, c3, stride=2, use_attention=use_attention),
            ResidualStage(c3, c3, stride=1, use_attention=use_attention),
            ResidualStage(c3, c4, stride=2, use_attention=use_attention),
            ResidualStage(c4, c4, stride=1, use_attention=use_attention),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.out_dim = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
