import numpy as np
import torch
def complex_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    计算复数余弦相似度
    Args:
        a, b: 复数张量，形状需广播一致
        dim:  计算相似度的维度
    Returns:
        余弦相似度（实数张量）
    """
    # 复数内积（对b取共轭）
    inner_product = torch.sum(a.conj() * b, dim=dim).abs()
    # 模长乘积
    norm_a = torch.linalg.norm(a.abs(), dim=dim)
    norm_b = torch.linalg.norm(b.abs(), dim=dim)
    return inner_product / (norm_a * norm_b + 1e-8)  # 防止除零
def complex_cosine_loss(pred, target):
    return 1 - complex_cosine_similarity(pred, target,dim=1)

def fnr(f1, f2, signal_dim):
    threshold = 1/(4*signal_dim)
    false_negative = np.zeros(f1.shape[0])
    nfreq = np.sum(f2 > -0.5, axis=1)
    for i in range(f2.shape[1]):
        dist_i_direct = np.min(np.abs(f1 - f2[:, i][:, None]), axis=1)
        dist_i_rshift = np.min(np.abs((f1 + 1) - f2[:, i][:, None]), axis=1)
        dist_i_lshift = np.min(np.abs((f1 - 1) - f2[:, i][:, None]), axis=1)
        dist_i = np.min((dist_i_direct, dist_i_rshift, dist_i_lshift), axis=0)
        valid_freq = (f2[:, i] != -10)
        false_negative += (dist_i > threshold)*valid_freq
    return np.sum(false_negative/nfreq)


def chamfer(f_estimate, f_target):
    f_estimate[f_estimate == -1] = -10.
    f_target[f_target == -1] = -10.
    dist_f_target = np.zeros(f_target.shape[0])
    for i in range(f_target.shape[1]):
        dist_i_direct = np.min(np.abs(f_estimate - f_target[:, i][:, None]), axis=1)
        dist_i_rshift = np.min(np.abs((f_estimate + 1) - f_target[:, i][:, None]), axis=1)
        dist_i_lshift = np.min(np.abs((f_estimate - 1) - f_target[:, i][:, None]), axis=1)
        dist_i = np.min((dist_i_direct, dist_i_rshift, dist_i_lshift), axis=0)
        b = (f_target[:, i] != -10.)
        dist_f_target += dist_i*b
    dist_f_estimate = np.zeros(f_estimate.shape[0])
    for i in range(f_estimate.shape[1]):
        dist_i_direct = np.min(np.abs(f_target - f_estimate[:, i][:, None]), axis=1)
        dist_i_rshift = np.min(np.abs((f_target + 1) - f_estimate[:, i][:, None]), axis=1)
        dist_i_lshift = np.min(np.abs((f_target - 1) - f_estimate[:, i][:, None]), axis=1)
        dist_i = np.min((dist_i_direct, dist_i_rshift, dist_i_lshift), axis=0)
        b = (f_estimate[:, i] != -10.)
        dist_f_estimate += dist_i*b
    dist = (dist_f_estimate + dist_f_target)
    return np.sum(dist)
