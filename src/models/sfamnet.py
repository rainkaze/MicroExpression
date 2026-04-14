import torch
import torch.nn as nn
from .blocks import CBAM


class SFAMNetLite(nn.Module):
    """
    SFAMNet-Lite: 针对 RGB 序列优化的双路光流注意力网络
    """

    def __init__(self, num_classes=4):
        super(SFAMNetLite, self).__init__()

        # 支路 1: 处理 U 分量 (水平肌肉运动)
        self.branch_u = self._make_branch()
        # 支路 2: 处理 V 分量 (垂直肌肉运动)
        self.branch_v = self._make_branch()

        # 特征融合与分类
        # 输入 128x128，经过 3 次 MaxPool 变为 16x16
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _make_branch(self):
        return nn.Sequential(
            # 第一层卷积和池化：提取纹理
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32),
            nn.MaxPool2d(2),

            # 第二层卷积和池化：提取局部
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.MaxPool2d(2),

            # 第三层卷积和池化：提取高层语义
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, u, v):
        feat_u = self.branch_u(u)
        feat_v = self.branch_v(v)

        # 融合两条路径的特征
        feat_u = self.avgpool(feat_u).view(feat_u.size(0), -1)
        feat_v = self.avgpool(feat_v).view(feat_v.size(0), -1)

        combined = torch.cat([feat_u, feat_v], dim=1)
        logits = self.classifier(combined)
        return logits