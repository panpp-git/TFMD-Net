import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ------------------------ 工具函数 ------------------------
def ensure_float32(tensor):
    """确保张量为float32类型"""
    if tensor.dtype == torch.float64:
        return tensor.float()
    return tensor


def ensure_complex64(tensor):
    """确保复数张量为complex64类型"""
    if tensor.dtype == torch.complex128:
        return tensor.to(torch.complex64)
    elif tensor.dtype in [torch.float32, torch.float64]:
        return tensor.to(torch.complex64)
    return tensor


def Differ5(y, delta):
    """五点差分计算导数 - 完全函数式"""
    L = y.shape[-1]
    if L < 3:
        return torch.zeros_like(y)

    # 构建差分结果，避免索引赋值
    left = (y[..., 1:2] - y[..., 0:1]) / delta
    middle = (y[..., 2:] - y[..., :-2]) / (2 * delta)
    right = (y[..., -1:] - y[..., -2:-1]) / delta

    return torch.cat([left, middle, right], dim=-1)


def projec5(vec, var):
    """投影操作，控制噪声 - 完全函数式"""
    if isinstance(var, (int, float)) and var == 0:
        return torch.zeros_like(vec)

    if vec.dim() == 1:
        M = vec.numel()
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.real.dtype, device=vec.device))
        n = torch.norm(torch.abs(vec))
        scale = torch.minimum(torch.ones_like(n), e / (n + 1e-12))
        return vec * scale
    else:
        M = vec.shape[-1]
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.real.dtype, device=vec.device))
        n = torch.norm(torch.abs(vec), dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(n), e / (n + 1e-12))
        return vec * scale


# ------------------------ 参数微调网络 ------------------------
class ParameterRefinement(nn.Module):
    """学习alpha和beta的微调值"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.refinement_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )

    def forward(self, base_alpha, base_beta, iteration=1):
        base_alpha = ensure_float32(base_alpha)
        base_beta = ensure_float32(base_beta)

        inputs = torch.stack([base_alpha, base_beta], dim=0).unsqueeze(0)
        refinement = self.refinement_net(inputs)

        alpha_delta = 0.3 * refinement[0, 0] * base_alpha
        beta_delta = 0.3 * refinement[0, 1] * base_beta

        refined_alpha = torch.clamp(base_alpha + alpha_delta, min=1e-6)
        refined_beta = torch.clamp(base_beta + beta_delta, min=1e-6)

        return refined_alpha, refined_beta

# ------------------------ 参数微调网络 ------------------------
class ParameterRefinement_improve(nn.Module):
    """学习 alpha 和 beta 的微调值，结合 init_if 信息"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        # 输入维度: (alpha, beta, init_if_stat)
        self.refinement_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )

    def forward(self, base_alpha, base_beta, init_if, iteration=1):
        base_alpha = ensure_float32(base_alpha)
        base_beta = ensure_float32(base_beta)

        # 统计 init_if 信息，比如全局均值 (可以换成 max, std 等)
        init_stat = init_if.abs().mean().unsqueeze(0)  # 标量

        # 拼接输入 (alpha, beta, init_if_stat)
        inputs = torch.stack([base_alpha, base_beta, init_stat.to(base_alpha.device)], dim=0).unsqueeze(0)

        refinement = self.refinement_net(inputs)

        # 计算微调
        alpha_delta = 0.3 * refinement[0, 0] * base_alpha
        beta_delta = 0.3 * refinement[0, 1] * base_beta

        refined_alpha = torch.clamp(base_alpha + alpha_delta, min=1e-6)
        refined_beta = torch.clamp(base_beta + beta_delta, min=1e-6)

        return refined_alpha, refined_beta



# ------------------------ 完全函数式的VNCMD网络层 ------------------------
class VNCMDLayerFunctional(nn.Module):
    """完全函数式的VNCMD迭代层，无任何就地操作"""

    def __init__(self, fs, use_parameter_refinement=True):
        super().__init__()
        self.fs = fs
        self.use_parameter_refinement = use_parameter_refinement

        if use_parameter_refinement:
            self.param_refiner = ParameterRefinement_improve()

    def forward(self, s, eIF, xm, ym, sum_x, sum_y, lamuda, alpha, beta, var,
                iteration=1, mode_mask=None):
        """
        完全函数式的VNCMD迭代，通过构建新张量避免所有就地操作
        """
        device = s.device

        # 确保输入数据类型正确
        s = ensure_complex64(s)
        eIF = ensure_float32(eIF)
        xm = ensure_complex64(xm)
        ym = ensure_complex64(ym)
        sum_x = ensure_complex64(sum_x)
        sum_y = ensure_complex64(sum_y)
        lamuda = ensure_complex64(lamuda)
        if mode_mask is not None:
            mode_mask = ensure_float32(mode_mask)

        batch_size, T = s.shape
        N = eIF.shape[1]

        # 参数微调
        current_alpha = alpha
        current_beta = beta
        if self.use_parameter_refinement:
            current_alpha, current_beta = self.param_refiner(alpha, beta, eIF,iteration)

        # 自适应学习率
        lr = 0.03 / (1 + iteration * 0.015)

        # 投影操作
        u_list = []
        for b in range(batch_size):
            u_b = projec5(s[b] - sum_x[b] - sum_y[b] - lamuda[b] / current_alpha, var)
            u_list.append(u_b)
        u = torch.stack(u_list, dim=0)

        # 为每个batch和模态创建新张量，完全避免索引赋值
        new_xm_list = []
        new_ym_list = []
        new_eIF_list = []
        new_sum_x_list = []
        new_sum_y_list = []

        for b in range(batch_size):
            batch_xm_list = []
            batch_ym_list = []
            batch_eIF_list = []
            batch_sum_x = torch.zeros(T, dtype=torch.complex64, device=device)
            batch_sum_y = torch.zeros(T, dtype=torch.complex64, device=device)

            for n in range(N):
                # 检查有效模态
                is_active = True
                if mode_mask is not None and mode_mask[b, n] == 0:
                    is_active = False
                if torch.allclose(eIF[b, n, :], torch.zeros_like(eIF[b, n, :]), atol=1e-6):
                    is_active = False

                if not is_active:
                    # 对于无效模态，保持原值
                    batch_xm_list.append(xm[b, n, :].clone())
                    batch_ym_list.append(ym[b, n, :].clone())
                    batch_eIF_list.append(eIF[b, n, :].clone())
                    continue

                # 计算当前模态的相位和三角函数
                phase = 2 * torch.pi * torch.cumsum(eIF[b, n, :], dim=0) / self.fs
                cosm = torch.cos(phase)
                sinm = torch.sin(phase)

                # 计算其他模态的累积贡献
                other_sum_x = torch.zeros(T, dtype=torch.complex64, device=device)
                other_sum_y = torch.zeros(T, dtype=torch.complex64, device=device)

                for other_n in range(N):
                    if other_n == n:
                        continue

                    other_is_active = True
                    if mode_mask is not None and mode_mask[b, other_n] == 0:
                        other_is_active = False
                    if torch.allclose(eIF[b, other_n, :], torch.zeros_like(eIF[b, other_n, :]), atol=1e-6):
                        other_is_active = False

                    if not other_is_active:
                        continue

                    other_phase = 2 * torch.pi * torch.cumsum(eIF[b, other_n, :], dim=0) / self.fs
                    other_cosm = torch.cos(other_phase)
                    other_sinm = torch.sin(other_phase)

                    other_sum_x = other_sum_x + xm[b, other_n, :] * other_cosm
                    other_sum_y = other_sum_y + ym[b, other_n, :] * other_sinm

                # 计算残差
                residual = s[b] - other_sum_x - other_sum_y - u[b] - lamuda[b] / current_alpha

                # 更新 xm, ym（完全函数式）
                xm_grad = cosm * residual.conj()
                ym_grad = sinm * residual.conj()

                # 添加平滑正则化（函数式）
                if T > 2:
                    xm_current = xm[b, n, :]
                    ym_current = ym[b, n, :]

                    # 计算二阶差分（函数式）
                    xm_diff = xm_current[2:] - 2 * xm_current[1:-1] + xm_current[:-2]
                    ym_diff = ym_current[2:] - 2 * ym_current[1:-1] + ym_current[:-2]

                    # 构建完整的平滑项
                    smooth_weight = 1e-4
                    xm_smooth_full = torch.cat([
                        torch.zeros(1, dtype=xm_grad.dtype, device=device),
                        -smooth_weight * xm_diff,
                        torch.zeros(1, dtype=xm_grad.dtype, device=device)
                    ], dim=0)
                    ym_smooth_full = torch.cat([
                        torch.zeros(1, dtype=ym_grad.dtype, device=device),
                        -smooth_weight * ym_diff,
                        torch.zeros(1, dtype=ym_grad.dtype, device=device)
                    ], dim=0)

                    xm_grad = xm_grad + xm_smooth_full
                    ym_grad = ym_grad + ym_smooth_full

                new_xm_n = xm[b, n, :] + lr * xm_grad
                new_ym_n = ym[b, n, :] + lr * ym_grad

                # 更新瞬时频率（完全函数式）
                xbar = Differ5(new_xm_n.real, 1 / self.fs)
                ybar = Differ5(new_ym_n.real, 1 / self.fs)
                den = new_xm_n.real ** 2 + new_ym_n.real ** 2 + 1e-12
                deltaIF = (new_xm_n.real * ybar - new_ym_n.real * xbar) / (2 * torch.pi * den)

                # IF平滑（函数式）
                if T > 2:
                    if_current = eIF[b, n, :]
                    if_diff = if_current[2:] - 2 * if_current[1:-1] + if_current[:-2]
                    if_smooth_full = torch.cat([
                        torch.zeros(1, dtype=deltaIF.dtype, device=device),
                        -1e-4 * if_diff,
                        torch.zeros(1, dtype=deltaIF.dtype, device=device)
                    ], dim=0)
                    deltaIF = deltaIF + if_smooth_full

                new_eIF_n = eIF[b, n, :] - lr * deltaIF
                # 约束IF范围
                new_eIF_n = torch.clamp(new_eIF_n, min=2.0, max=self.fs / 2 - 2)

                # 添加到列表
                batch_xm_list.append(new_xm_n)
                batch_ym_list.append(new_ym_n)
                batch_eIF_list.append(new_eIF_n)

                # 累积贡献
                new_phase = 2 * torch.pi * torch.cumsum(new_eIF_n, dim=0) / self.fs
                new_cosm = torch.cos(new_phase)
                new_sinm = torch.sin(new_phase)

                batch_sum_x = batch_sum_x + new_xm_n * new_cosm
                batch_sum_y = batch_sum_y + new_ym_n * new_sinm

            # 使用stack构建批次张量
            new_xm_list.append(torch.stack(batch_xm_list, dim=0))
            new_ym_list.append(torch.stack(batch_ym_list, dim=0))
            new_eIF_list.append(torch.stack(batch_eIF_list, dim=0))
            new_sum_x_list.append(batch_sum_x)
            new_sum_y_list.append(batch_sum_y)

        # 构建最终输出张量
        new_xm = torch.stack(new_xm_list, dim=0)
        new_ym = torch.stack(new_ym_list, dim=0)
        new_eIF = torch.stack(new_eIF_list, dim=0)
        new_sum_x = torch.stack(new_sum_x_list, dim=0)
        new_sum_y = torch.stack(new_sum_y_list, dim=0)

        # 更新拉格朗日乘数
        new_lamuda = lamuda + current_alpha * (u + new_sum_x + new_sum_y - s)

        return new_eIF, new_xm, new_ym, new_sum_x, new_sum_y, new_lamuda


# ------------------------ VNCMD 完整网络 ------------------------
class VNCMDNet(nn.Module):
    """完全函数式的深度展开VNCMD网络"""

    def __init__(self, fs, max_layers=30, use_parameter_refinement=True):
        super().__init__()
        self.fs = fs
        self.max_layers = max_layers
        self.use_parameter_refinement = use_parameter_refinement

        # 可学习的全局参数
        self.alpha = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # 使用完全函数式的网络层
        self.layers = nn.ModuleList([
            VNCMDLayerFunctional(fs, use_parameter_refinement)
            for _ in range(max_layers)
        ])

        print(f"创建完全函数式VNCMD网络: {max_layers} 层")

    def _detect_active_modes(self, eIF):
        """检测有效模态（非零padding）"""
        batch_size, N, T = eIF.shape
        mode_mask_list = []

        for b in range(batch_size):
            batch_mask_list = []
            for n in range(N):
                if not torch.allclose(eIF[b, n, :], torch.zeros_like(eIF[b, n, :]), atol=1e-6):
                    batch_mask_list.append(torch.tensor(1.0, device=eIF.device))
                else:
                    batch_mask_list.append(torch.tensor(0.0, device=eIF.device))
            mode_mask_list.append(torch.stack(batch_mask_list))

        return torch.stack(mode_mask_list)

    def forward(self, s, eIF, var=0.0, num_iterations=None, mode_mask=None,
                tol=1e-7, return_all_iterations=False):
        """前向传播 - 完全函数式"""
        device = s.device

        s = ensure_complex64(s)
        eIF = ensure_float32(eIF)
        if mode_mask is not None:
            mode_mask = ensure_float32(mode_mask)

        batch_size, T = s.shape
        N = eIF.shape[1]

        if mode_mask is None:
            mode_mask = self._detect_active_modes(eIF)

        # 初始化 - 使用列表和stack构建
        xm_list = []
        ym_list = []
        sum_x_list = []
        sum_y_list = []

        for b in range(batch_size):
            batch_xm_list = []
            batch_ym_list = []
            batch_sum_x = torch.zeros(T, dtype=torch.complex64, device=device)
            batch_sum_y = torch.zeros(T, dtype=torch.complex64, device=device)

            for n in range(N):
                if mode_mask[b, n] == 0:
                    batch_xm_list.append(torch.zeros(T, dtype=torch.complex64, device=device))
                    batch_ym_list.append(torch.zeros(T, dtype=torch.complex64, device=device))
                    continue

                phase = 2 * torch.pi * torch.cumsum(eIF[b, n, :], dim=0) / self.fs
                sinm_bn = torch.sin(phase)
                cosm_bn = torch.cos(phase)

                xm_bn = s[b].real * cosm_bn + s[b].imag * sinm_bn
                ym_bn = s[b].real * sinm_bn - s[b].imag * cosm_bn

                batch_xm_list.append(xm_bn)
                batch_ym_list.append(ym_bn)

                batch_sum_x = batch_sum_x + xm_bn * cosm_bn
                batch_sum_y = batch_sum_y + ym_bn * sinm_bn

            xm_list.append(torch.stack(batch_xm_list, dim=0))
            ym_list.append(torch.stack(batch_ym_list, dim=0))
            sum_x_list.append(batch_sum_x)
            sum_y_list.append(batch_sum_y)

        xm = torch.stack(xm_list, dim=0)
        ym = torch.stack(ym_list, dim=0)
        sum_x = torch.stack(sum_x_list, dim=0)
        sum_y = torch.stack(sum_y_list, dim=0)
        lamuda = torch.zeros((batch_size, T), dtype=torch.complex64, device=device)

        # 存储迭代结果
        if return_all_iterations:
            all_eIF = [eIF.detach().clone()]
            all_xm = [xm.detach().clone()]
            all_ym = [ym.detach().clone()]

        # 迭代求解
        max_iter = num_iterations if num_iterations is not None else self.max_layers
        iteration = 0

        for layer_idx in range(min(max_iter, self.max_layers)):
            old_eIF = eIF.detach().clone() if num_iterations is None else None

            eIF, xm, ym, sum_x, sum_y, lamuda = self.layers[layer_idx](
                s, eIF, xm, ym, sum_x, sum_y, lamuda,
                self.alpha, self.beta, var, layer_idx + 1, mode_mask
            )

            iteration += 1

            if return_all_iterations:
                all_eIF.append(eIF.detach().clone())
                all_xm.append(xm.detach().clone())
                all_ym.append(ym.detach().clone())

            # 收敛检查
            if num_iterations is None and old_eIF is not None:
                sDif_list = []
                for b in range(batch_size):
                    for n in range(N):
                        if mode_mask[b, n] == 1:
                            diff_norm = torch.norm(eIF[b, n, :] - old_eIF[b, n, :])
                            old_norm = torch.norm(old_eIF[b, n, :])
                            sDif_list.append((diff_norm / (old_norm + 1e-12)) ** 2)

                if len(sDif_list) > 0:
                    sDif = torch.sqrt(torch.mean(torch.stack(sDif_list)))
                    if sDif.item() < tol:
                        break

        # 计算最终结果
        IA = torch.sqrt(xm.real ** 2 + ym.real ** 2)

        # 重构信号 - 函数式
        reconstructed_list = []
        for b in range(batch_size):
            batch_recon = torch.zeros(T, dtype=torch.complex64, device=device)
            for n in range(N):
                if mode_mask[b, n] == 0:
                    continue

                phase = 2 * torch.pi * torch.cumsum(eIF[b, n, :], dim=0) / self.fs
                cosm_recon = torch.cos(phase)
                sinm_recon = torch.sin(phase)
                mode_contrib = xm[b, n, :] * cosm_recon + ym[b, n, :] * sinm_recon
                batch_recon = batch_recon + mode_contrib

            reconstructed_list.append(batch_recon)

        reconstructed = torch.stack(reconstructed_list, dim=0)

        result = {
            'eIF': eIF,
            'IA': IA,
            'reconstructed': reconstructed,
            'iterations': iteration,
            'xm': xm,
            'ym': ym,
            'mode_mask': mode_mask
        }

        if return_all_iterations:
            result.update({
                'all_eIF': all_eIF,
                'all_xm': all_xm,
                'all_ym': all_ym
            })

        return result


# ------------------------ 损失函数 ------------------------
class VNCMDLoss(nn.Module):
    """VNCMD网络的损失函数"""

    def __init__(self, lambda_recon=1.0, lambda_if=0.5, lambda_smooth=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_if = lambda_if
        self.lambda_smooth = lambda_smooth

    def forward(self, result, target_signal, target_if=None):
        mode_mask = result['mode_mask']

        target_signal = ensure_complex64(target_signal)
        if target_if is not None:
            target_if = ensure_float32(target_if)

        # 重构损失
        recon_loss = F.mse_loss(result['reconstructed'].real, target_signal.real) + \
                     F.mse_loss(result['reconstructed'].imag, target_signal.imag)

        total_loss = self.lambda_recon * recon_loss
        loss_dict = {'recon_loss': recon_loss}

        # IF损失
        if target_if is not None:
            valid_mask = mode_mask.unsqueeze(-1).expand_as(target_if)
            if_loss = F.mse_loss(result['eIF'] * valid_mask, target_if * valid_mask)
            total_loss += self.lambda_if * if_loss
            loss_dict['if_loss'] = if_loss

        # 平滑损失
        if self.lambda_smooth > 0:
            smooth_loss_list = []
            batch_size, N = mode_mask.shape

            for b in range(batch_size):
                for n in range(N):
                    if mode_mask[b, n] == 1:
                        if_diff = torch.diff(result['eIF'][b, n, :])
                        smooth_loss_list.append(torch.mean(if_diff ** 2))

            if len(smooth_loss_list) > 0:
                smooth_loss = torch.mean(torch.stack(smooth_loss_list))
                total_loss += self.lambda_smooth * smooth_loss
                loss_dict['smooth_loss'] = smooth_loss

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


# ------------------------ 数据生成器 ------------------------
class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, fs=256, duration=1.0, max_modes=3):
        self.num_samples = num_samples
        self.fs = fs
        self.duration = duration
        self.max_modes = max_modes
        self.T = int(fs * duration)
        self.t = torch.arange(0, duration, 1 / fs, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_modes = torch.randint(1, self.max_modes + 1, (1,)).item()

        modes = []
        ifs = []

        for k in range(actual_modes):
            f_start = torch.rand(1) * 50 + 25  # 25-75 Hz
            f_end = torch.rand(1) * 50 + 25
            f_if = f_start + (f_end - f_start) * self.t / self.duration

            amp = torch.rand(1) * 0.5 + 0.5  # 0.5-1.0
            phase = 2 * torch.pi * torch.cumsum(f_if, dim=0) / self.fs
            mode = amp * torch.exp(1j * phase)
            mode = ensure_complex64(mode)

            modes.append(mode)
            ifs.append(f_if)

        signal = sum(modes)
        signal = ensure_complex64(signal)

        # 添加噪声
        noise_level = torch.rand(1) * 0.02
        noise = noise_level * (torch.randn_like(signal.real) + 1j * torch.randn_like(signal.imag))
        signal = signal + ensure_complex64(noise)

        # 填充
        while len(ifs) < self.max_modes:
            ifs.append(torch.zeros_like(self.t))
            modes.append(torch.zeros_like(signal))

        modes = [ensure_complex64(mode) for mode in modes]
        ifs = [ensure_float32(if_) for if_ in ifs]

        true_if = torch.stack(ifs, dim=0)
        init_if = true_if.clone()
        for k in range(actual_modes):
            init_if[k, :] += torch.randn_like(init_if[k, :]) * 1.5
            init_if[k, :] = torch.clamp(init_if[k, :], min=5.0, max=self.fs / 2 - 5)

        mode_mask = torch.zeros(self.max_modes, dtype=torch.float32)
        mode_mask[:actual_modes] = 1.0

        return {
            'signal': signal,
            'true_if': true_if,
            'init_if': init_if,
            'mode_mask': mode_mask,
            'actual_modes': actual_modes,
            'individual_modes': torch.stack(modes, dim=0)
        }


# ------------------------ 训练函数 ------------------------
def train_vncmd_net():
    """训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    fs = 256
    batch_size = 2  # 小批次提高稳定性
    num_epochs = 10
    learning_rate = 1e-4
    max_modes = 3

    train_dataset = SignalDataset(num_samples=100, fs=fs, max_modes=max_modes)
    val_dataset = SignalDataset(num_samples=20, fs=fs, max_modes=max_modes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    net = VNCMDNet(fs, max_layers=15, use_parameter_refinement=True).to(device)
    criterion = VNCMDLoss(lambda_recon=1.0, lambda_if=0.3, lambda_smooth=0.05)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)

    train_losses = []
    val_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        net.train()
        epoch_train_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                signal = ensure_complex64(batch['signal']).to(device)
                true_if = ensure_float32(batch['true_if']).to(device)
                init_if = ensure_float32(batch['init_if']).to(device)
                mode_mask = ensure_float32(batch['mode_mask']).to(device)

                optimizer.zero_grad()

                result = net(signal, init_if, var=0.005, num_iterations=6, mode_mask=mode_mask)
                loss, loss_dict = criterion(result, signal, true_if)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
                else:
                    print(f'跳过batch {batch_idx}: loss={loss.item()}')

            except Exception as e:
                print(f"训练batch {batch_idx}时出错: {str(e)}")
                continue

        if num_batches == 0:
            print(f"Epoch {epoch}: 没有成功的训练batch")
            continue

        # 验证
        net.eval()
        epoch_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    signal = ensure_complex64(batch['signal']).to(device)
                    true_if = ensure_float32(batch['true_if']).to(device)
                    init_if = ensure_float32(batch['init_if']).to(device)
                    mode_mask = ensure_float32(batch['mode_mask']).to(device)

                    result = net(signal, init_if, var=0.005, num_iterations=8, mode_mask=mode_mask)
                    loss, _ = criterion(result, signal, true_if)

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        epoch_val_loss += loss.item()
                        val_batches += 1

                except Exception as e:
                    continue

        if val_batches > 0:
            train_losses.append(epoch_train_loss / num_batches)
            val_losses.append(epoch_val_loss / val_batches)

            print(f'Epoch {epoch}: Train Loss {train_losses[-1]:.6f}, Val Loss {val_losses[-1]:.6f}')
            print(f'Alpha={net.alpha.item():.6f}, Beta={net.beta.item():.6f}')

    return net, train_losses, val_losses


# ------------------------ 测试函数 ------------------------
def test_vncmd_net():
    """测试函数"""
    print("=== 测试完全函数式VNCMD网络 ===")

    fs = 256
    t = torch.arange(0, 1, 1 / fs, dtype=torch.float32)

    f1 = 40 + 20 * t
    f2 = 80 - 15 * t

    phase1 = 2 * torch.pi * torch.cumsum(f1, dim=0) / fs
    phase2 = 2 * torch.pi * torch.cumsum(f2, dim=0) / fs

    s1 = 0.6 * torch.exp(1j * phase1)
    s2 = 0.4 * torch.exp(1j * phase2)
    s = s1 + s2

    s = ensure_complex64(s)
    f1 = ensure_float32(f1)
    f2 = ensure_float32(f2)

    max_modes = 3
    T = len(t)

    eIF = torch.zeros((max_modes, T), dtype=torch.float32)
    eIF[0, :] = f1 + torch.randn_like(f1) * 2.0
    eIF[1, :] = f2 + torch.randn_like(f2) * 2.0
    eIF = torch.clamp(eIF, min=5.0, max=fs / 2 - 5)

    s_batch = s.unsqueeze(0)
    eIF_batch = eIF.unsqueeze(0)

    net = VNCMDNet(fs, max_layers=20, use_parameter_refinement=True)

    print("\n=== 测试不同迭代次数 ===")
    for num_iter in [8, 12, 16]:
        print(f"\n--- 使用 {num_iter} 次迭代 ---")
        with torch.no_grad():
            result = net(s_batch, eIF_batch, var=0.0, num_iterations=num_iter)

        print(f"实际迭代次数: {result['iterations']}")
        print(f"检测到的有效模态: {torch.sum(result['mode_mask'])}")

        true_ifs = torch.stack([f1, f2, torch.zeros_like(f1)])
        for k in range(2):
            error = torch.mean((result['eIF'][0, k, :] - true_ifs[k]) ** 2)
            print(f"模态 {k + 1} IF误差: {error:.6f}")

        recon_error = F.mse_loss(result['reconstructed'][0].real, s.real) + \
                      F.mse_loss(result['reconstructed'][0].imag, s.imag)
        print(f"重构误差: {recon_error:.6f}")

    # 可视化
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, s.real, label='Original', linewidth=2)
    plt.plot(t, result['reconstructed'][0].real, '--', label='Reconstructed', linewidth=2)
    plt.legend()
    plt.title('Signal Reconstruction')

    plt.subplot(2, 2, 2)
    plt.plot(t, f1, '--', label='True IF 1', alpha=0.7, linewidth=2)
    plt.plot(t, result['eIF'][0, 0, :], label='Est IF 1', linewidth=1.5)
    plt.plot(t, f2, '--', label='True IF 2', alpha=0.7, linewidth=2)
    plt.plot(t, result['eIF'][0, 1, :], label='Est IF 2', linewidth=1.5)
    plt.legend()
    plt.title('IF Estimation')

    plt.subplot(2, 2, 3)
    plt.plot(t, torch.abs(s1), '--', label='True IA 1', alpha=0.7, linewidth=2)
    plt.plot(t, result['IA'][0, 0, :], label='Est IA 1', linewidth=1.5)
    plt.plot(t, torch.abs(s2), '--', label='True IA 2', alpha=0.7, linewidth=2)
    plt.plot(t, result['IA'][0, 1, :], label='Est IA 2', linewidth=1.5)
    plt.legend()
    plt.title('IA Estimation')

    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.7, f'Alpha: {net.alpha.item():.6f}', fontsize=12)
    plt.text(0.1, 0.5, f'Beta: {net.beta.item():.6f}', fontsize=12)
    plt.text(0.1, 0.3, f'Iterations: {result["iterations"]}', fontsize=12)
    plt.title('Network Parameters')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return net, result


if __name__ == "__main__":
    print("完全函数式VNCMD网络")
    print("特点: 无任何就地操作，纯函数式构建")

    choice = input("\n选择模式 (1-测试, 2-训练): ").strip()

    if choice == "2":
        train_vncmd_net()
    else:
        test_vncmd_net()