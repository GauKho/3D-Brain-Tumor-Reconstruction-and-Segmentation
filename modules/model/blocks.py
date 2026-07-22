"""Reusable 3D building blocks for LATUPNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, *_ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1, 1)
        return x * weights


class ParallelConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shared_conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.branches = nn.ModuleList(
            [
                nn.Conv3d(out_channels, out_channels, 1),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.Conv3d(out_channels, out_channels, 5, padding=2),
            ]
        )
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward_with_skip(self, x: torch.Tensor):
        skip = self.activation(self.shared_conv(x))
        paths = [self.pool(self.activation(branch(skip))) for branch in self.branches]
        return skip, torch.cat(paths, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_skip(x)[1]


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_se=True, dropout_rate=0.2):
        super().__init__()
        self.se = SqueezeExcitation(in_channels) if use_se else nn.Identity()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.se(x)
        x = self.activation(self.norm1(self.conv1(x)))
        skip = self.dropout(self.activation(self.norm2(self.conv2(x))))
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_se=True, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.se(self.activation(self.norm1(self.conv1(x))))
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.dropout(self.activation(self.norm2(self.conv2(x))))
