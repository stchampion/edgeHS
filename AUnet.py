"""
Attention-gated U-Net for heart sound spectrogram denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(g_channels, out_channels, kernel_size=1)
        self.W_x = nn.Conv2d(x_channels, out_channels, kernel_size=1)
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


def up_conv(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )


class AUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AUNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, 16, dropout_rate=0.5)
        self.enc2 = ConvBlock(16, 32, dropout_rate=0.5)
        self.enc3 = ConvBlock(32, 64, dropout_rate=0.5)
        self.enc4 = ConvBlock(64, 128, dropout_rate=0.5)
        self.bottleneck = ConvBlock(128, 256, dropout_rate=0.5)

        # Decoder
        self.up4 = up_conv(256, 128)
        self.att4 = AttentionGate(128, 128, 64)
        self.dec4 = ConvBlock(256, 128, dropout_rate=0.3)

        self.up3 = up_conv(128, 64)
        self.att3 = AttentionGate(64, 64, 32)
        self.dec3 = ConvBlock(128, 64, dropout_rate=0.3)

        self.up2 = up_conv(64, 32)
        self.att2 = AttentionGate(32, 32, 16)
        self.dec2 = ConvBlock(64, 32, dropout_rate=0.3)

        self.up1 = up_conv(32, 16)
        self.att1 = AttentionGate(16, 16, 8)
        self.dec1 = ConvBlock(32, 16, dropout_rate=0.3)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Crop from 129 to 128 to match paper
        if x.shape[2] == 129:
            x = x[:, :, :128, :]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention gates
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    model = AUNet()
    x = torch.randn(1, 1, 129, 64)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")