import torch
import torch.nn as nn
from .Conv import ParallelConvolutionBlock
from .SqueezeExcitation import SqueezeExcitation

class EncoderBlock(nn.Module):
    """Encoder block with SE attention, two convolutions, and dropout"""
    def __init__(self, in_channels, out_channels, use_se=True, dropout_rate=0.2):
        super().__init__()
        self.se = SqueezeExcitation(in_channels) if use_se else nn.Identity()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu  = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        # Paper không đề cập InstanceNorm sau conv2 trong encoder, nhưng Table 1 có
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        self.dropout = nn.Dropout3d(dropout_rate)
        self.pool    = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.se(x)                            # SE trước
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        skip = x                                   # skip trước pooling
        x = self.pool(x)
        return x, skip