import torch
import torch.nn as nn
import torch.nn.functional as F
from .Conv import ParallelConvolutionBlock
from .SqueezeExcitation import SqueezeExcitation
from .Encoder import EncoderBlock
from .Decoder import DecoderBlock

class LATUPNet(nn.Module):
    """
    Lightweight 3D Attention U-Net with Parallel Convolutions (LATUP-Net)
    
    Based on the paper:
    "LATUP-Net: A lightweight 3D attention U-Net with parallel convolutions 
    for brain tumor segmentation" - Alwadee et al., Computers in Biology and Medicine 2025
    """
    
    def __init__(self, in_channels=3, num_classes=3, use_se=True, dropout_rate=0.2):
        super().__init__()

        # PC block: (B,3,128,128,128) → (B,96,64,64,64)
        self.pc_block = ParallelConvolutionBlock(in_channels, 32)

        # Encoder 2: (B,96,64,64,64) → (B,64,32,32,32), skip1: (B,64,64,64,64)
        self.enc2 = EncoderBlock(96, 64, use_se=use_se, dropout_rate=dropout_rate)
        # Encoder 3: (B,64,32,32,32) → (B,128,16,16,16), skip2: (B,128,32,32,32)
        self.enc3 = EncoderBlock(64, 128, use_se=use_se, dropout_rate=dropout_rate)

        # Bottleneck SE (Table 1: SE Layer_3 at bottleneck)
        self.bottleneck_se = SqueezeExcitation(128) if use_se else nn.Identity()

        # Decoder 3: (B,128,16,16,16) + skip2(B,128,32,32,32) → (B,128,32,32,32)
        self.dec3 = DecoderBlock(128, 128, 128, use_se=use_se, dropout_rate=dropout_rate)
        # Decoder 2: (B,128,32,32,32) + skip1(B,64,64,64,64) — wait, skip1 là 64ch
        # Theo Table 1: dec2 concat shape (64,64,64,128) → out 64
        self.dec2 = DecoderBlock(128, 64, 64, use_se=use_se, dropout_rate=dropout_rate)
        # Decoder 1: (B,64,64,64,64) upsample → (B,128,128,128) + skip_pc
        # skip từ PC block output (B,96,64,64,64) — nhưng paper dùng skip từ PC trước pool
        # Table 1 dec1_concat shape: (128,128,128,64) → paper dùng skip của PC embedded (32ch trước pool)
        # dec1: conv1 64→32, concat skip_pc(32ch per path? không) 
        # Theo Table 1 chính xác: dec1_conv1 input=64→32, concat→64ch, conv2 64→32
        self.dec1 = DecoderBlock(64, 32, 32, use_se=False, dropout_rate=dropout_rate)

        # Final 1x1x1 conv → softmax (paper Section 3)
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # PC block — lưu skip từ shared conv (trước pool) theo Table 1 dec1_concat
        # Table 1: dec1_concat input (128,128,128,32) → skip là output shared_conv (32ch, full res)
        # Tuy nhiên paper Fig.1 cho thấy skip từ PC là toàn bộ PC output trước pool
        # Theo Table 1 dec1_concat shape=(128,128,128,64): 32(dec1_conv1 out) + 32(skip_pc shared) = 64 ✓
        x_shared = self.pc_block.relu(self.pc_block.shared_conv(x))  # (B,32,128,128,128)
        skip_pc  = x_shared

        x = self.pc_block.pool(self.pc_block.relu(self.pc_block.conv1(x_shared)))
        p2 = self.pc_block.pool(self.pc_block.relu(self.pc_block.conv3(x_shared)))
        p3 = self.pc_block.pool(self.pc_block.relu(self.pc_block.conv5(x_shared)))
        x = torch.cat([x, p2, p3], dim=1)       # (B,96,64,64,64)

        x, skip2 = self.enc2(x)                  # x:(B,64,32,32,32), skip2:(B,64,64,64,64)
        x, skip3 = self.enc3(x)                  # x:(B,128,16,16,16), skip3:(B,128,32,32,32)

        x = self.bottleneck_se(x)                # (B,128,16,16,16)

        x = self.dec3(x, skip3)                  # (B,128,32,32,32)
        x = self.dec2(x, skip2)                  # (B,64,64,64,64)
        x = self.dec1(x, skip_pc)                # (B,32,128,128,128)

        return self.final_conv(x)                 # (B,4,128,128,128) — full res, no extra upsample needed