import torch
import torch.nn as nn

class ParallelConvolutionBlock(nn.Module):
    """
    PC Block theo paper Fig.2:
    shared 3x3x3 embed → 3 parallel paths (1x1x1, 3x3x3, 5x5x5) → maxpool → concat
    Không có BatchNorm (paper không đề cập norm trong PC block)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shared_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x_shared = self.relu(self.shared_conv(x))
        path1 = self.pool(self.relu(self.conv1(x_shared)))
        path2 = self.pool(self.relu(self.conv3(x_shared)))
        path3 = self.pool(self.relu(self.conv5(x_shared)))
        return torch.cat([path1, path2, path3], dim=1)