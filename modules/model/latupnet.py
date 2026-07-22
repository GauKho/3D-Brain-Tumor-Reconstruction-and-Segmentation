"""Nine-channel LATUPNet architecture from the wavelet notebook."""

import torch
import torch.nn as nn

from .blocks import DecoderBlock, EncoderBlock, ParallelConvolutionBlock, SqueezeExcitation


class LATUPNet(nn.Module):
    """Segment 3 MRI and 6 wavelet channels into WT, TC and ET regions."""

    def __init__(self, in_channels=9, num_classes=3, use_se=True, dropout_rate=0.2):
        super().__init__()
        self.pc_block = ParallelConvolutionBlock(in_channels, 32)
        self.enc2 = EncoderBlock(96, 64, use_se, dropout_rate)
        self.enc3 = EncoderBlock(64, 128, use_se, dropout_rate)
        self.bottleneck_se = SqueezeExcitation(128) if use_se else nn.Identity()
        self.dec3 = DecoderBlock(128, 128, 128, use_se, dropout_rate)
        self.dec2 = DecoderBlock(128, 64, 64, use_se, dropout_rate)
        self.dec1 = DecoderBlock(64, 32, 32, False, dropout_rate)
        self.final_conv = nn.Conv3d(32, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_pc, x = self.pc_block.forward_with_skip(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x = self.dec3(self.bottleneck_se(x), skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip_pc)
        return self.final_conv(x)
