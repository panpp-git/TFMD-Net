import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = sample_gumbel(logits.size())
    y = logits + gumbel_noise.to(logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        shape = y.size()
        _, max_idx = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0).to(logits.device)
        y = (y_hard - y).detach() + y

    return y


# logits = torch.tensor([1.2, 0.9, 2.5])
# temperature = 0.5
#
# samples = gumbel_softmax(logits, temperature, hard=True)
# print(samples)
#
# # 我们有一个简单的损失函数
# target = torch.tensor([0, 0, 1])  # 假设真实类别是第三类
# loss = F.cross_entropy(samples.unsqueeze(0), target.unsqueeze(0).argmax(dim=-1))
#
# # 反向传播
# loss.backward()  # 这里是可行的，因为Gumbel-Softmax是可微的