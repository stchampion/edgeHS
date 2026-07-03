"""
Lightweight CNN classifier for heart sound abnormality detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        avg_pool = self.fc1(avg_pool)
        avg_pool = F.relu(avg_pool)
        avg_pool = self.fc2(avg_pool)
        avg_pool = self.sigmoid(avg_pool).view(avg_pool.size(0), avg_pool.size(1), 1, 1)
        return x * avg_pool


class MultiScaleDepthwise(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(MultiScaleDepthwise, self).__init__()
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                             padding=1, groups=in_channels, bias=False)
        self.dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride,
                             padding=2, groups=in_channels, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn5 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out3 = self.dw3(x)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)

        out5 = self.dw5(x)
        out5 = self.bn5(out5)
        out5 = self.relu(out5)

        out = torch.cat([out3, out5], dim=1)
        out = self.fusion(out)
        return out


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_multiscale=False):
        super(DepthwiseSeparableBlock, self).__init__()

        if use_multiscale:
            self.depthwise = MultiScaleDepthwise(in_channels, stride=stride)
        else:
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                       padding=1, groups=in_channels, bias=False)
            self.depthwise_bn = nn.BatchNorm2d(in_channels)
            self.depthwise_relu = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_relu = nn.ReLU(inplace=True)

        self.se = SEBlock(out_channels)
        self.use_multiscale = use_multiscale

    def forward(self, x):
        if self.use_multiscale:
            x = self.depthwise(x)
        else:
            x = self.depthwise(x)
            x = self.depthwise_bn(x)
            x = self.depthwise_relu(x)

        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_relu(x)

        x = self.se(x)
        return x


class LightCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LightCNN, self).__init__()

        self.block1 = DepthwiseSeparableBlock(1, 16, stride=2, use_multiscale=False)
        self.block2 = DepthwiseSeparableBlock(16, 32, stride=2, use_multiscale=False)
        self.block3 = DepthwiseSeparableBlock(32, 64, stride=2, use_multiscale=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = LightCNN(num_classes=2)
    x = torch.randn(1, 1, 128, 64)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")