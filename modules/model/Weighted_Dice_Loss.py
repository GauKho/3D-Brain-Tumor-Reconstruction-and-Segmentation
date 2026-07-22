import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedDiceLoss(nn.Module):
    """Weighted Dice Score Loss (WDL) as defined in the paper"""
    
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.w_wt = 1.64
        self.w_tc = 2.55
        self.w_et = 3.40

    def _dice_loss(self, pred, target):
        pred_f   = pred.contiguous().view(-1)
        target_f = target.contiguous().view(-1)
        intersection = (pred_f * target_f).sum()
        return 1.0 - (2. * intersection + self.epsilon) / (
            pred_f.pow(2).sum() + target_f.pow(2).sum() + self.epsilon
        )

    def forward(self, pred, target):
        """
        pred:   (B, 4, H, W, D) — logits, channels: [BG, NCR/NET, ED, ET]
        target: (B, H, W, D)    — labels {0,1,2,4} (BraTS convention)
        """
        pred_soft = F.softmax(pred, dim=1)

        # One-hot targets theo channel BraTS: label4 → channel3
        target_ncr = (target == 1).float()
        target_ed  = (target == 2).float()
        target_et  = (target == 4).float()

        dsl_ncr = self._dice_loss(pred_soft[:, 1], target_ncr)
        dsl_ed  = self._dice_loss(pred_soft[:, 2], target_ed)
        dsl_et  = self._dice_loss(pred_soft[:, 3], target_et)

        # Eq.(4) paper
        wdl = (self.w_wt + self.w_tc + self.w_et) * dsl_et \
            + (self.w_wt + self.w_tc)              * dsl_ncr \
            + self.w_wt                             * dsl_ed

        return wdl
