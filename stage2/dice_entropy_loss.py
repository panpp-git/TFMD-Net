import torch
import torch.nn  as nn
import torch.nn.functional  as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.2, bce_weight=0.8, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = 0.25
        self.gamma = 2

    def forward(self, pred, target,wgt=None):
        # 输入检查：pred为(B,1,H,W)，target为(B,H,W)或(B,1,H,W)
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B,H,W) → (B,1,H,W)
        target = target.float()


        # Focalloss计算
        if wgt is None:
            bce_loss = torch.mean(self.bce(pred, target))
        else:
            bce_loss = torch.mean(wgt.unsqueeze(1) * self.bce(pred, target))
        # pt = torch.exp(-bce_loss)  # p_t = exp(-BCE_loss)
        # 计算 Focal Loss
        # F_loss = torch.mean(self.alpha * (1 - pt) ** self.gamma * bce_loss)

        # Dice计算（直接基于logits+Sigmoid）
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        dice_loss = dice_loss.mean()


        return self.dice_weight * dice_loss + self.bce_weight * bce_loss,dice_loss*self.dice_weight
# class BinaryDiceBCELoss(nn.Module):
#     def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
#         super().__init__()
#         self.dice_weight = dice_weight
#         self.bce_weight = bce_weight
#         self.smooth = smooth
#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, pred, target):
#         # 输入检查：pred为(B,1,H,W)，target为(B,H,W)或(B,1,H,W)
#         if target.dim() == 3:
#             target = target.unsqueeze(1)  # (B,H,W) → (B,1,H,W)
#         target = target.float()

#         # BCE计算
#         bce_loss = self.bce(pred, target)

#         # Dice计算（直接基于logits+Sigmoid）
#         pred_prob = torch.sigmoid(pred)
#         intersection = (pred_prob * target).sum(dim=(2, 3))
#         union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
#         dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
#         dice_loss_eval=dice_loss.sum()
#         dice_loss = dice_loss.mean()


#         return self.dice_weight * dice_loss + self.bce_weight * bce_loss,dice_loss_eval


if __name__ == "__main__":
    # 测试数据生成
    batch_size, H, W = 2, 128, 128
    num_classes = 1  # 二分类（前景/背景）

    # 模型预测（未经激活的logits）
    pred = torch.randn(batch_size, num_classes, H, W)
    # 真实标签（二分类时可以是0/1或概率）
    target = torch.randint(0, 2, (batch_size, H, W)).long()

    # 损失计算
    combined_loss = BinaryDiceBCELoss()

    print("Combined Loss:", combined_loss(pred, target))
