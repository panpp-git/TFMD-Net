import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class SimpleConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 四层卷积生成多尺度特征
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),  # 输出尺寸：(B,64,H/2,W/2)
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),  # 输出尺寸：(B,128,H/4,W/4)
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),  # 输出尺寸：(B,256,H/8,W/8)
            nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )  # 输出尺寸：(B,512,H/16,W/16)
        ])

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features  # 返回4级特征图列表


class InstanceMaskHead(nn.Module):
    def __init__(self, num_queries=10, hidden_dim=256):
        super().__init__()
        self.num_queries = num_queries
        # 可学习查询向量（每个代表一个实例）
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Transformer解码器
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=8,
                dim_feedforward=2048, batch_first=True
            ),
            num_layers=3
        )
        # 掩码生成器
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1)
        )

    def forward(self, features):
        # 取最深层的特征进行解码
        x = features[-1]  # (B,512,H/16,W/16)
        B, C, H, W = x.shape
        # 特征展平
        flattened_feature = x.flatten(2)  # (B, H*W/256, 512)
        # 查询向量生成
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,10,256)
        # Transformer解码
        decoded = self.transformer(queries, flattened_feature)  # (B,10,256)
        # 生成掩码
        masks = decoded.unsqueeze(-1).unsqueeze(-1)  # (B,10,256,1,1)
        masks = self.mask_conv((masks * x.unsqueeze(1)).view(20,256,16,16)).view(2,10,16,16)  # (B,10,1,H/16,W/16)
        masks = F.interpolate(masks, scale_factor=16, mode='bilinear')  # (B,10,1,H,W)
        return torch.sigmoid(masks)  # (B,10,H,W)


class InstanceMask2Former(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleConvBackbone()
        self.head = InstanceMaskHead()

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    # 生成模拟数据


# batch_size = 2
# input_tensor = torch.randn(batch_size, 3, 256, 256)  # 输入尺寸：(2,3,256,256)
#
# # 模型初始化
# model = InstanceMask2Former()
# output = model(input_tensor)
#
# # 输出尺寸验证
# print("输入尺寸:", input_tensor.shape)  # torch.Size([2, 3, 256, 256])
# print("输出尺寸:", output.shape)  # torch.Size([2, 10, 256, 256])
# # 生成模拟标签（假设每张图最多5个实例）
# gt_masks = torch.randint(0, 2, (batch_size, 5, 256, 256)).float()


# 计算匈牙利匹配损失
def hungarian_loss(pred, target):
    # 简化的Dice损失计算
    cost_matrix = 1 - (2 * (pred[:, None] * target[:, :, None]).sum((3, 4)) + 1e-6) / \
                  (pred[:, None].sum((3, 4)) + target[:, :, None].sum((3, 4)) + 1e-6)
    cost_matrix=torch.chunk(cost_matrix,2,dim=0)
    indices = [linear_sum_assignment(cost.squeeze(0).cpu().detach().numpy()) for cost in cost_matrix]
    loss = sum(F.binary_cross_entropy(pred[i, idx[1]], target[i, idx[0]]) for i, idx in enumerate(indices))
    return loss / 2

# # 梯度回传验证
# loss = hungarian_loss(output, gt_masks)
# loss.backward()
#
# # 检查梯度存在性
# print("骨干网络梯度:", model.backbone.layers[0][0].weight.grad.shape)  # torch.Size([64, 3, 3, 3])
# print("查询向量梯度:", model.head.query_embed.weight.grad.shape)  # torch.Size([10, 256])