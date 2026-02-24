import torch.nn as nn
import torch
import numpy as np

import torch
import numpy as np
import torch.nn.functional as F

def generate_random_sample(img_size=128, max_objects=2):
    # 生成随机图像（模拟输入）
    image = torch.rand(3, img_size, img_size)  # 3通道

    # 生成标签（最多2个目标）
    masks = torch.zeros((max_objects, img_size, img_size))  # 每个目标的mask
    classes = torch.zeros(max_objects, dtype=torch.long)  # 类别标签（0为背景，1/2为目标类）

    # 随机生成矩形和圆形作为目标
    for i in range(max_objects):
        if np.random.rand() > 0.5:  # 50%概率生成目标
            # 随机生成矩形或圆形
            x, y = np.random.randint(20, 100, 2)
            w, h = np.random.randint(10, 30, 2)
            if np.random.rand() > 0.5:
                # 矩形mask
                masks[i, y:y + h, x:x + w] = 1
            else:
                # 圆形mask
                rr, cc = np.ogrid[y - 10:y + 10, x - 10:x + 10]
                mask = (rr - y) ** 2 + (cc - x) ** 2 <= 10 ** 2
                masks[i,y - 10:y + 10,x - 10:x + 10] = torch.from_numpy(mask)
        classes[i] = torch.tensor(np.random.randint(1, 3) ) # 类别1或2

    return image, {"masks": masks, "classes": classes}


# 调试输出尺寸
image, targets = generate_random_sample()
print("Image shape:", image.shape)  # (3, 128, 128)
print("Masks shape:", targets["masks"].shape)  # (2, 128, 128)
print("Classes:", targets["classes"])  # 如 tensor([1, 0])




class SegmentationNet(nn.Module):
    def __init__(self, num_classes=2, num_queries=10):
        super().__init__()
        # 编码器（ResNet简化版）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16
        )

        # 解码器分支
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, num_queries, kernel_size=1)  # 10x64x64
        )

        # 分类分支
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Flatten(),
            nn.Linear(128, num_queries * (num_classes + 1))  # 10个查询，每类+背景
        )

    def forward(self, x):
        features = self.encoder(x)  # (B,128,16,16)
        mask_pred = self.mask_head(features)  # (B,10,64,64)
        class_logits = self.class_head(features)  # (B,10*(2+1))
        class_logits = class_logits.view(-1, 10, 3)  # (B,10,3)
        mask_pred=torch.nn.Parameter(torch.rand(1,10,128,128))
        # class_logits=torch.rand((1,10,3))
        return {"masks": mask_pred, "classes": class_logits}

net = SegmentationNet()
outputs = net(image.unsqueeze(0))

print("Mask outputs:", outputs["masks"].shape)    # (1,10,64,64)
print("Class outputs:", outputs["classes"].shape) # (1,10,3)


class DifferentiableMatcher(nn.Module):
    def __init__(self, num_iters=5, temperature=0.1):
        super().__init__()
        self.num_iters = num_iters
        self.temp = temperature

    def forward(self, pred_masks, gt_masks):
        # 计算代价矩阵（mask IoU）
        gt_masks=gt_masks.unsqueeze(0)
        cost = 1 - torch.einsum('bkhw,bnhw->bkn', pred_masks, gt_masks)
        # cost = 1 - torch.mean(pred_masks.unsqueeze(2) * gt_masks.unsqueeze(1), dim=(3, 4))

        # Sinkhorn迭代生成软分配矩阵
        log_alpha = -cost / self.temp
        for _ in range(self.num_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        soft_assignment = torch.exp(log_alpha)

        return soft_assignment  # (B,10,2)


def compute_loss(pred, target, matcher):
    # 匹配预测与真实标签
    assignment = matcher(pred["masks"], target["masks"])

    # 分类损失（交叉熵）
    matched_classes = torch.einsum("bqg,bg->bq", assignment, target["classes"].float().unsqueeze(0))
    cls_loss = F.cross_entropy(pred["classes"].argmax(dim=2).to(torch.float), matched_classes)

    # Mask损失（Dice系数）
    matched_masks = torch.einsum("bqg,bglhw->bqlhw", assignment, target["masks"].unsqueeze(0).unsqueeze(2)).squeeze(2)
    mask_loss = 1 - (2 * (pred["masks"] * matched_masks).sum() + 1e-6) / (pred["masks"].sum() + matched_masks.sum() + 1e-6)

    return cls_loss + mask_loss


# 梯度检查
image.requires_grad_(True)
outputs = net(image.unsqueeze(0))
loss = compute_loss(outputs, targets, DifferentiableMatcher())
loss.backward()
print("梯度存在:", image.grad is not None)  # 应输出True

# 简单训练循环
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
for _ in range(10):  # 示例迭代
    image, targets = generate_random_sample()
    outputs = net(image.unsqueeze(0))
    loss = compute_loss(outputs, targets, DifferentiableMatcher())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")