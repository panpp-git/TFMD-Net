# VNCMD网络训练完整示例
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


# ------------------------ 工具函数 ------------------------
def differ5_torch(y, delta):
    """五点差分计算导数"""
    L = y.shape[-1]
    ybar = torch.zeros_like(y)
    if L >= 3:
        ybar[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / (2 * delta)
        ybar[..., 0] = (y[..., 1] - y[..., 0]) / delta
        ybar[..., -1] = (y[..., -1] - y[..., -2]) / delta
    return ybar


def cumtrapz_torch(y, dx):
    """累积梯形积分，保持长度一致"""
    cumsum = torch.zeros_like(y)
    if y.shape[-1] > 1:
        cumsum[..., 1:] = torch.cumsum((y[..., :-1] + y[..., 1:]) * 0.5 * dx, dim=-1)
    return cumsum


def projec5(vec, var):
    """投影操作，控制噪声"""
    if isinstance(var, (int, float)) and var == 0:
        return torch.zeros_like(vec)

    # 支持批量处理
    if vec.dim() == 1:
        M = vec.numel()
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.dtype, device=vec.device))
        n = torch.norm(vec)
        if n > e:
            return vec * (e / n)
        else:
            return vec
    else:
        # 批量处理
        M = vec.shape[-1]
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.dtype, device=vec.device))
        n = torch.norm(vec, dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(n), e / (n + 1e-12))
        return vec * scale


def build_second_diff_matrix(N, device, dtype=torch.float32):
    """构建二阶差分矩阵"""
    e = torch.ones(N, dtype=dtype, device=device)
    e2 = -2.0 * torch.ones(N, dtype=dtype, device=device)
    e2[0] = -1.0
    e2[-1] = -1.0
    oper = torch.diag(e2) + torch.diag(e[:-1], -1) + torch.diag(e[:-1], 1)
    opedoub = oper.T @ oper
    return opedoub


# ------------------------ 超参数学习网络 ------------------------
class HyperparameterRefinement(nn.Module):
    """基于初始频率和当前超参数学习残差项的网络"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # 频率特征提取器
        self.freq_encoder = nn.Sequential(
            nn.Linear(2, 32),  # 输入平均频率
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # 超参数残差预测器
        self.param_refiner = nn.Sequential(
            nn.Linear(16 + 2, hidden_dim),  # 16(频率特征) + 2(当前alpha,beta)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 输出alpha和beta的残差
            nn.Tanh()  # 限制残差范围
        )

        # 迭代自适应权重
        self.iteration_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, init_freqs, current_alpha, current_beta, iteration=1):
        """
        根据初始频率和当前超参数预测残差

        Args:
            init_freqs: (batch_size, K) 初始平均频率
            current_alpha, current_beta: 当前超参数值
            iteration: 当前迭代次数
        """
        batch_size, K = init_freqs.shape

        # 提取频率特征 - 使用平均频率作为代表
        # avg_freqs = torch.mean(init_freqs, dim=1, keepdim=True)  # (batch_size, 1)
        freq_features = self.freq_encoder(init_freqs)  # (batch_size, 16)

        # 当前超参数
        current_params = torch.stack([
            current_alpha.expand(batch_size),
            current_beta.expand(batch_size)
        ], dim=1)  # (batch_size, 2)

        # 拼接特征
        combined_features = torch.cat([freq_features, current_params], dim=1)

        # 预测残差
        residuals = self.param_refiner(combined_features)  # (batch_size, 2)

        # 应用迭代自适应权重
        iteration_factor = torch.sigmoid(self.iteration_weight * iteration)
        residuals = residuals * iteration_factor * 0.1  # 控制残差幅度

        # 计算新的超参数
        alpha_residual = residuals[:, 0] * current_alpha
        beta_residual = residuals[:, 1] * current_beta

        new_alpha = torch.clamp(current_alpha + alpha_residual.mean(), min=1e-6, max=1e-2)
        new_beta = torch.clamp(current_beta + beta_residual.mean(), min=1e-6, max=1e-1)

        return new_alpha, new_beta


# ------------------------ 可微分线性求解器 ------------------------
class DifferentiableLinearSolver(nn.Module):
    """可微分的线性系统求解器，避免torch.linalg.solve的梯度问题"""

    def __init__(self, max_iter=50, tol=1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, A, b, x_init=None):
        """
        使用共轭梯度法求解 Ax = b

        Args:
            A: (N, N) 正定对称矩阵
            b: (N,) 右端向量
            x_init: (N,) 初始解（可选）
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype

        # 添加正则化保证正定性
        reg = 1e-6 * torch.eye(N, device=device, dtype=dtype)
        A_reg = A + reg

        # 初始化
        if x_init is None:
            x = torch.zeros_like(b)
        else:
            x = x_init.clone()

        r = b - torch.mv(A_reg, x)
        p = r.clone()
        rsold = torch.dot(r, r)

        # 共轭梯度迭代
        for i in range(self.max_iter):
            Ap = torch.mv(A_reg, p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)

            if torch.sqrt(rsnew) < self.tol:
                break

            beta = rsnew / (rsold + 1e-12)
            p = r + beta * p
            rsold = rsnew

        return x


# ------------------------ VNCMD网络层 ------------------------
class VNCMDLayer(nn.Module):
    """单个VNCMD迭代层"""

    def __init__(self, use_hyperparameter_learning=True):
        super().__init__()
        self.use_hyperparameter_learning = use_hyperparameter_learning

        if use_hyperparameter_learning:
            self.hyperparam_refiner = HyperparameterRefinement()

        self.linear_solver = DifferentiableLinearSolver(max_iter=30)

    def forward(self, s, eIF, xm, ym, sum_x, sum_y, lamuda,
                alpha, beta, var, fs, iteration=1, mode_mask=None):
        """
        单步VNCMD迭代 (list→stack 版本，避免 inplace)
        """
        device = s.device
        batch_size, N = s.shape
        K = eIF.shape[1]
        dtype = s.dtype

        # 计算初始平均频率（用于超参数学习）
        init_freqs = torch.zeros((batch_size, K,2), device=device, dtype=s.dtype)
        for b in range(batch_size):
            for k in range(K):
                if mode_mask[b, k] > 0:
                    init_freqs[b, k,0] = torch.mean(torch.diff(eIF[b, k, :]))
                    init_freqs[b, k, 1] = torch.var(torch.diff(eIF[b, k, :]))
        init_freqs=torch.mean(init_freqs,dim=1,keepdim=True)
        init_freqs=torch.flatten(init_freqs,1)
        # 超参数学习
        current_alpha = alpha
        current_beta = beta
        if self.use_hyperparameter_learning and init_freqs is not None:
            current_alpha, current_beta = self.hyperparam_refiner(
                init_freqs, alpha, beta, iteration
            )

        # 动态 beta 调整
        betathr = torch.minimum(
            (10 ** (iteration / 36.0 - 10.0)) * torch.ones_like(current_beta),
            current_beta
        )

        # 二阶差分矩阵
        opedoub = build_second_diff_matrix(N, device, dtype)
        inv_fs = torch.tensor(1.0 / fs, device=device, dtype=dtype)
        eyeN = torch.eye(N, device=device, dtype=dtype)

        # ====== 批量处理 ======
        new_eIF_batches = []
        new_xm_batches = []
        new_ym_batches = []
        new_sum_x_batches = []
        new_sum_y_batches = []
        new_lamuda_batches = []

        for b in range(batch_size):
            # 投影
            u_b = projec5(s[b] - sum_x[b] - sum_y[b] - lamuda[b] / current_alpha, var)

            # 累积量
            batch_sum_x = torch.zeros(N, device=device, dtype=dtype)
            batch_sum_y = torch.zeros(N, device=device, dtype=dtype)

            xm_list = []
            ym_list = []
            eif_list = []

            for k in range(K):
                if mode_mask is not None and mode_mask[b, k] == 0:
                    xm_list.append(xm[b, k, :])
                    ym_list.append(ym[b, k, :])
                    eif_list.append(eIF[b, k, :])
                    continue

                # 去除旧贡献
                temp_sum_x = sum_x[b] - xm[b, k, :] * torch.cos(
                    2 * math.pi * cumtrapz_torch(eIF[b, k, :], inv_fs)
                )
                temp_sum_y = sum_y[b] - ym[b, k, :] * torch.sin(
                    2 * math.pi * cumtrapz_torch(eIF[b, k, :], inv_fs)
                )

                # 相位
                phase = 2 * math.pi * cumtrapz_torch(eIF[b, k, :], inv_fs)
                cosm_k = torch.cos(phase)
                sinm_k = torch.sin(phase)

                # 更新 xm
                A_x = (2.0 / current_alpha) * opedoub + torch.diag(cosm_k ** 2)
                rhs_x = cosm_k * (s[b] - temp_sum_x - temp_sum_y - u_b - lamuda[b] / current_alpha)
                solved_x = self.linear_solver(A_x, rhs_x, xm[b, k, :])

                # 更新 ym
                A_y = (2.0 / current_alpha) * opedoub + torch.diag(sinm_k ** 2)
                rhs_y = sinm_k * (s[b] - temp_sum_x - temp_sum_y - u_b - lamuda[b] / current_alpha)
                solved_y = self.linear_solver(A_y, rhs_y, ym[b, k, :])

                # IF 更新
                xbar = differ5_torch(solved_x, inv_fs)
                ybar = differ5_torch(solved_y, inv_fs)
                denom = solved_x ** 2 + solved_y ** 2 + 1e-12
                deltaIF = (solved_x * ybar - solved_y * xbar) / (denom * 2 * math.pi)

                S = (2.0 / betathr) * opedoub + eyeN
                deltaIF_smooth = self.linear_solver(S, deltaIF, torch.zeros_like(deltaIF))
                new_eif_k = eIF[b, k, :] - 0.5 * deltaIF_smooth

                # 新相位
                new_phase = 2 * math.pi * cumtrapz_torch(new_eif_k, inv_fs)
                new_cosm = torch.cos(new_phase)
                new_sinm = torch.sin(new_phase)

                batch_sum_x = batch_sum_x + solved_x * new_cosm
                batch_sum_y = batch_sum_y + solved_y * new_sinm

                xm_list.append(solved_x)
                ym_list.append(solved_y)
                eif_list.append(new_eif_k)

            # 拼接该样本的结果
            new_xm_batches.append(torch.stack(xm_list, dim=0))
            new_ym_batches.append(torch.stack(ym_list, dim=0))
            new_eIF_batches.append(torch.stack(eif_list, dim=0))
            new_sum_x_batches.append(batch_sum_x)
            new_sum_y_batches.append(batch_sum_y)
            new_lamuda_batches.append(lamuda[b] + current_alpha * (u_b + batch_sum_x + batch_sum_y - s[b]))

        # ====== 拼接 batch 维度 ======
        new_eIF = torch.stack(new_eIF_batches, dim=0)
        new_xm = torch.stack(new_xm_batches, dim=0)
        new_ym = torch.stack(new_ym_batches, dim=0)
        new_sum_x = torch.stack(new_sum_x_batches, dim=0)
        new_sum_y = torch.stack(new_sum_y_batches, dim=0)
        new_lamuda = torch.stack(new_lamuda_batches, dim=0)

        return new_eIF, new_xm, new_ym, new_sum_x, new_sum_y, new_lamuda, current_alpha, current_beta




# ------------------------ 深度展开VNCMD网络 ------------------------
class DeepUnfoldedVNCMD(nn.Module):
    """深度展开的VNCMD网络"""

    def __init__(self, max_layers=50, use_hyperparameter_learning=False):
        super().__init__()
        self.max_layers = max_layers
        self.use_hyperparameter_learning = use_hyperparameter_learning

        # 全局可学习超参数
        self.global_alpha = nn.Parameter(torch.tensor(3e-4, dtype=torch.float32))
        self.global_beta = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

        # 网络层
        self.layers = nn.ModuleList([
            VNCMDLayer(use_hyperparameter_learning)
            for _ in range(max_layers)
        ])

        # print(f"创建深度展开VNCMD网络: {max_layers} 层, 超参数学习: {use_hyperparameter_learning}")

    def _detect_active_modes(self, eIF):
        """检测有效模态"""
        batch_size, K, N = eIF.shape
        mode_mask = torch.zeros((batch_size, K), device=eIF.device, dtype=eIF.dtype)

        for b in range(batch_size):
            for k in range(K):
                if not torch.allclose(eIF[b, k, :], torch.zeros_like(eIF[b, k, :]), atol=1e-6):
                    mode_mask[b, k] = 1.0

        return mode_mask

    def forward(self, s, eIF, fs, var=0.0, num_iterations=None, mode_mask=None,
                tol=1e-7, return_history=False):
        """
        前向传播

        Args:
            s: (batch_size, N) 输入信号
            eIF: (batch_size, K, N) 初始瞬时频率
            fs: 采样频率
            var: 噪声方差
            num_iterations: 迭代次数
            mode_mask: (batch_size, K) 模态掩码
            tol: 收敛容差
            return_history: 是否返回历史记录
        """
        device = s.device

        # 处理单信号输入
        squeeze_output = False
        if s.dim() == 1:
            s = s.unsqueeze(0)
            eIF = eIF.unsqueeze(0)
            if mode_mask is not None:
                mode_mask = mode_mask.unsqueeze(0)
            squeeze_output = True

        batch_size, N = s.shape
        K = eIF.shape[1]

        # 自动检测有效模态
        if mode_mask is None:
            mode_mask = self._detect_active_modes(eIF)



        # 初始化模态分量
        xm = torch.zeros((batch_size, K, N), device=device, dtype=s.dtype)
        ym = torch.zeros((batch_size, K, N), device=device, dtype=s.dtype)
        sum_x = torch.zeros((batch_size, N), device=device, dtype=s.dtype)
        sum_y = torch.zeros((batch_size, N), device=device, dtype=s.dtype)
        lamuda = torch.zeros((batch_size, N), device=device, dtype=s.dtype)

        # 初始化各模态
        opedoub = build_second_diff_matrix(N, device, s.dtype)
        solver = DifferentiableLinearSolver()

        for b in range(batch_size):
            batch_sum_x = torch.zeros(N, device=device, dtype=s.dtype)
            batch_sum_y = torch.zeros(N, device=device, dtype=s.dtype)

            for k in range(K):
                if mode_mask[b, k] == 0:
                    continue

                phase = 2 * math.pi * cumtrapz_torch(eIF[b, k, :], torch.tensor(1 / fs))
                cosm = torch.cos(phase)
                sinm = torch.sin(phase)

                # 初始化xm, ym
                A_x = (2.0 / self.global_alpha) * opedoub + torch.diag(cosm ** 2)
                A_y = (2.0 / self.global_alpha) * opedoub + torch.diag(sinm ** 2)

                xm[b, k, :] = solver(A_x, cosm * s[b])
                ym[b, k, :] = solver(A_y, sinm * s[b])

                batch_sum_x += xm[b, k, :] * cosm
                batch_sum_y += ym[b, k, :] * sinm

            sum_x[b] = batch_sum_x
            sum_y[b] = batch_sum_y

        # 历史记录
        if return_history:
            eIF_history = [eIF.clone()]
            alpha_history = [self.global_alpha.clone()]
            beta_history = [self.global_beta.clone()]

        # 迭代优化
        max_iter = num_iterations if num_iterations is not None else self.max_layers
        iteration = 0
        current_alpha = self.global_alpha
        current_beta = self.global_beta

        for layer_idx in range(min(max_iter, self.max_layers)):
            old_eIF = eIF.clone()

            # 通过当前层
            eIF, xm, ym, sum_x, sum_y, lamuda, current_alpha, current_beta = self.layers[layer_idx](
                s, eIF, xm, ym, sum_x, sum_y, lamuda,
                current_alpha, current_beta, var, fs, layer_idx + 1, mode_mask
            )

            iteration += 1

            if return_history:
                eIF_history.append(eIF.clone())
                alpha_history.append(current_alpha.clone())
                beta_history.append(current_beta.clone())

            # 收敛检查
            if num_iterations is None:
                sDif = torch.tensor(0.0, device=device)
                valid_modes = 0

                for b in range(batch_size):
                    for k in range(K):
                        if mode_mask[b, k] > 0:
                            diff_norm = torch.norm(eIF[b, k, :] - old_eIF[b, k, :])
                            old_norm = torch.norm(old_eIF[b, k, :])
                            sDif += (diff_norm / (old_norm + 1e-12)) ** 2
                            valid_modes += 1

                if valid_modes > 0:
                    sDif = torch.sqrt(sDif / valid_modes)
                    if sDif.item() < tol:
                        break

        # 计算最终结果
        IA = torch.sqrt(xm ** 2 + ym ** 2)

        # 重构信号和各模态
        reconstructed = torch.zeros_like(s)
        modes = torch.zeros_like(xm)

        for b in range(batch_size):
            for k in range(K):
                if mode_mask[b, k] == 0:
                    continue

                phase = 2 * math.pi * cumtrapz_torch(eIF[b, k, :], torch.tensor(1 / fs))
                cosm = torch.cos(phase)
                sinm = torch.sin(phase)
                mode_signal = xm[b, k, :] * cosm + ym[b, k, :] * sinm
                modes[b, k, :] = mode_signal
                reconstructed[b] += mode_signal

        # 构建结果字典
        result = {
            'eIF': eIF,
            'IA': IA,
            'reconstructed': reconstructed,
            'modes': modes,
            'iterations': iteration,
            'final_alpha': current_alpha,
            'final_beta': current_beta,
            'mode_mask': mode_mask
        }

        if return_history:
            result.update({
                'eIF_history': eIF_history,
                'alpha_history': alpha_history,
                'beta_history': beta_history
            })

        # 处理单信号输出
        if squeeze_output:
            for key in ['eIF', 'IA', 'reconstructed', 'modes']:
                if key in result:
                    result[key] = result[key].squeeze(0)

        return result


# ------------------------ 损失函数 ------------------------
class VNCMDLoss(nn.Module):
    """VNCMD网络损失函数"""

    def __init__(self, lambda_recon=1.0, lambda_if=0.5, lambda_smooth=0.1, lambda_param=0.01):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_if = lambda_if
        self.lambda_smooth = lambda_smooth
        self.lambda_param = lambda_param

    def forward(self, result, target_signal, target_if=None):
        """计算损失"""
        mode_mask = result['mode_mask']

        # 重构损失
        recon_loss = F.mse_loss(result['reconstructed'], target_signal)
        total_loss = self.lambda_recon * recon_loss
        loss_dict = {'recon_loss': recon_loss}
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

def train_vncmd_network():
    """完整的VNCMD网络训练示例"""

    # 1. 创建训练数据
    fs = 256
    t = torch.arange(0, 1, 1 / fs, dtype=torch.float32)
    N = len(t)

    # 生成多模态信号
    f1 = 80 + 30 * t  # 线性调频
    f2 = 50 - 20 * t

    phase1 = 2 * torch.pi * torch.cumsum(f1, dim=0) / fs
    phase2 = 2 * torch.pi * torch.cumsum(f2, dim=0) / fs

    s1 = torch.sin(phase1)
    s2 = 0.8 * torch.sin(phase2)
    clean_signal = s1 + s2

    # 添加噪声
    noise_level = 0.1
    noisy_signal = clean_signal + noise_level * torch.randn_like(clean_signal)

    # 初始频率估计（带误差）
    init_if = torch.stack([
        f1 + torch.randn_like(f1) * 5,
        f2 + torch.randn_like(f2) * 5
    ], dim=0)

    target_if = torch.stack([f1, f2], dim=0)

    # 2. 创建网络和优化组件
    net = DeepUnfoldedVNCMD(
        max_layers=10,
        use_hyperparameter_learning=True
    )

    criterion = VNCMDLoss(
        lambda_recon=1.0,  # 重构损失权重
        lambda_if=0.5,  # 频率损失权重
        lambda_smooth=0.1,  # 平滑损失权重
        lambda_param=0.01  # 参数正则化权重
    )

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"网络参数数量: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    print(f"初始超参数: alpha={net.global_alpha.item():.6f}, beta={net.global_beta.item():.6f}")

    # 3. 训练循环
    # num_epochs = 20
    # best_loss = float('inf')
    # loss_history = []
    #
    # net.train()  # 训练模式
    #
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #
    #     # 前向传播（训练时使用较少迭代）
    #     result = net(
    #         noisy_signal,
    #         init_if,
    #         fs,
    #         var=noise_level ** 2,
    #         num_iterations=6  # 训练时少迭代，加快速度
    #     )
    #
    #     # 计算损失
    #     loss, loss_dict = criterion(result, clean_signal, target_if)
    #
    #     # 反向传播
    #     with torch.autograd.set_detect_anomaly(True):
    #         loss.backward()
    #
    #     # 梯度裁剪（防止梯度爆炸）
    #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    #
    #     # 参数更新
    #     optimizer.step()
    #     scheduler.step(loss)
    #
    #     # 记录
    #     loss_history.append(loss.item())
    #
    #     # 保存最佳模型
    #     if loss.item() < best_loss:
    #         best_loss = loss.item()
    #         torch.save({
    #             'model_state_dict': net.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss.item(),
    #             'epoch': epoch,
    #         }, 'best_vncmd_model.pth')
    #
    #     # 打印进度
    #     if epoch % 5 == 0:
    #         print(f"Epoch {epoch:3d}: "
    #               f"Loss={loss.item():.6f}, "
    #               f"Recon={loss_dict['recon_loss'].item():.4f}, "
    #               f"IF={loss_dict.get('if_loss', torch.tensor(0)).item():.4f}, "
    #               f"LR={optimizer.param_groups[0]['lr']:.2e}")
    #
    # print(f"训练完成! 最佳损失: {best_loss:.6f}")

    # 4. 测试阶段（更多迭代获得更好结果）
    net.eval()
    with torch.no_grad():
        test_result = net(
            noisy_signal,
            init_if,
            fs,
            var=noise_level ** 2,
            num_iterations=20  # 测试时多迭代，获得更好结果
        )

        test_loss, _ = criterion(test_result, clean_signal, target_if)

        print(f"\n测试结果:")
        print(f"  测试损失: {test_loss.item():.6f}")
        print(f"  实际迭代: {test_result['iterations']}")
        print(f"  最终alpha: {test_result['final_alpha'].item():.6f}")
        print(f"  最终beta: {test_result['final_beta'].item():.6f}")

        # 计算性能指标
        recon_error = torch.mean((test_result['reconstructed'] - clean_signal) ** 2)
        if1_error = torch.mean((test_result['eIF'][0, :] - f1) ** 2)
        if2_error = torch.mean((test_result['eIF'][1, :] - f2) ** 2)

        print(f"  重构MSE: {recon_error.item():.6f}")
        print(f"  IF1 MSE: {if1_error.item():.6f}")
        print(f"  IF2 MSE: {if2_error.item():.6f}")

    return net, test_result


def batch_training_example():
    """批量训练示例"""
    print("\n=== 批量训练示例 ===")

    # 生成批量数据
    fs = 128
    batch_size = 8
    t = torch.arange(0, 0.8, 1 / fs, dtype=torch.float32)
    N = len(t)

    signals = []
    init_ifs = []
    target_ifs = []

    for i in range(batch_size):
        # 每个样本不同的参数
        f1 = 60 + 10 * i + 20 * t
        f2 = 40 - 5 * i + 15 * t

        phase1 = 2 * torch.pi * torch.cumsum(f1, dim=0) / fs
        phase2 = 2 * torch.pi * torch.cumsum(f2, dim=0) / fs

        s = torch.sin(phase1) + 0.7 * torch.sin(phase2)
        s += 0.05 * torch.randn_like(s)  # 添加噪声

        init_if = torch.stack([f1 + torch.randn_like(f1) * 3,
                               f2 + torch.randn_like(f2) * 3])
        target_if = torch.stack([f1, f2])

        signals.append(s)
        init_ifs.append(init_if)
        target_ifs.append(target_if)

    # 转换为批量张量
    batch_signals = torch.stack(signals)  # (batch_size, N)
    batch_init_ifs = torch.stack(init_ifs)  # (batch_size, K, N)
    batch_target_ifs = torch.stack(target_ifs)  # (batch_size, K, N)

    # 网络和训练设置
    net = DeepUnfoldedVNCMD(max_layers=8, use_hyperparameter_learning=True)
    criterion = VNCMDLoss(lambda_recon=1.0, lambda_if=0.3, lambda_smooth=0.05)
    optimizer = optim.Adam(net.parameters(), lr=5e-4)

    # 批量训练
    net.train()
    for epoch in range(15):
        optimizer.zero_grad()

        # 批量前向传播
        batch_result = net(batch_signals, batch_init_ifs, fs,
                           var=0.05 ** 2, num_iterations=5)

        # 批量损失计算
        batch_loss, _ = criterion(batch_result, batch_signals, batch_target_ifs)

        # 反向传播
        batch_loss.backward()

        # 梯度裁剪和参数更新
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 3 == 0:
            print(f"Batch Epoch {epoch}: Loss = {batch_loss.item():.6f}")

    print("批量训练完成!")
    return net


# 使用示例
if __name__ == "__main__":

    fs = 256
    t = torch.arange(0, 1, 1 / fs, dtype=torch.float32)
    N = len(t)

    f1 = 150 - 80 * t  # mode 1 IF
    f2 = 50 + 60 * t  # mode 2 IF

    phase1 = 2 * math.pi * torch.cumsum(f1, dim=0) / fs
    phase2 = 2 * math.pi * torch.cumsum(f2, dim=0) / fs
    s1 = torch.sin(phase1)
    s2 = torch.sin(phase2)
    s = s1 + s2

    noise_level = 0.1
    noise = noise_level * torch.randn_like(s)
    s_noisy = s + noise

    eIF_init = torch.stack([
        f1 + torch.randn_like(f1) * 5,
        f2 + torch.randn_like(f2) * 5
    ], dim=0)

    net = DeepUnfoldedVNCMD(max_layers=30, use_hyperparameter_learning=False)

    print(f"alpha={net.global_alpha.item():.6f}, beta={net.global_beta.item():.6f}")

    result = net(s_noisy, eIF_init, fs, var=noise_level ** 2, num_iterations=20)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(t, s, 'b-', label='Clean Signal', linewidth=2)
    plt.plot(t, s_noisy, 'k-', label='Noisy Signal', alpha=0.7)
    plt.plot(t, result['reconstructed'].detach().cpu().numpy(), 'r--', label='Reconstructed', linewidth=2)
    plt.title('Signal Reconstruction')
    plt.legend()
    plt.grid(True)

    # IFä¼°è®¡
    plt.subplot(2, 3, 2)
    plt.plot(t, f1, 'b--', label='True IF1', alpha=0.7, linewidth=2)
    plt.plot(t, result['eIF'][0, :].detach().cpu().numpy(), 'b-', label='Estimated IF1', linewidth=2)
    plt.plot(t, f2, 'g--', label='True IF2', alpha=0.7, linewidth=2)
    plt.plot(t, result['eIF'][1, :].detach().cpu().numpy(), 'g-', label='Estimated IF2', linewidth=2)
    plt.title('IF Estimation')
    plt.legend()
    plt.grid(True)

    # æ¨¡æ€åˆ†ç¦»
    plt.subplot(2, 3, 3)
    plt.plot(t, s1, 'b--', label='True Mode 1', alpha=0.7, linewidth=2)
    plt.plot(t, result['modes'][0, :].detach().cpu().numpy(), 'b-', label='Estimated Mode 1', linewidth=2)
    plt.title('Mode 1 Separation')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(t, s2, 'g--', label='True Mode 2', alpha=0.7, linewidth=2)
    plt.plot(t, result['modes'][1, :].detach().cpu().numpy(), 'g-', label='Estimated Mode 2', linewidth=2)
    plt.title('Mode 2 Separation')
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 3, 5)
    plt.plot(t, result['IA'][0, :].detach().cpu().numpy(), 'b-', label='IA Mode 1', linewidth=2)
    plt.plot(t, result['IA'][1, :].detach().cpu().numpy(), 'g-', label='IA Mode 2', linewidth=2)
    plt.title('Instantaneous Amplitudes')
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'Iterations: {result["iterations"]}', fontsize=12)
    plt.text(0.1, 0.6, f'Final Alpha: {result["final_alpha"]:.6f}', fontsize=12)
    plt.text(0.1, 0.4, f'Final Beta: {result["final_beta"]:.6f}', fontsize=12)
    plt.title('Network Info')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#############################################################################################
    print("VNCMD网络训练演示")
    print("=" * 40)
    # 单样本训练
    net, result = train_vncmd_network()
    # 批量训练
    batch_net = batch_training_example()

    print("\n🎉 所有训练测试完成!")
    print("网络完全支持梯度反向传播和端到端训练!")