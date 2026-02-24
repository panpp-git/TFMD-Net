import torch
import matplotlib.pyplot as plt
import numpy as np


# ------------------------ 工具函数 ------------------------
def Differ5(y, delta):
    """五点差分计算导数"""
    L = y.shape[-1]
    ybar = torch.zeros_like(y)
    if L >= 3:
        ybar[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / (2 * delta)
        ybar[..., 0] = (y[..., 1] - y[..., 0]) / delta
        ybar[..., -1] = (y[..., -1] - y[..., -2]) / delta
    return ybar


def projec5(vec, var):
    """投影操作，控制噪声"""
    if isinstance(var, (int, float)) and var == 0:
        return torch.zeros_like(vec)

    if vec.dim() == 1:
        # 单个向量
        M = vec.numel()
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.real.dtype, device=vec.device))
        n = torch.norm(torch.abs(vec))
        if n > e:
            return vec * (e / n)
        else:
            return vec
    else:
        # 批量处理 (batch_size, T)
        M = vec.shape[-1]
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.real.dtype, device=vec.device))
        n = torch.norm(torch.abs(vec), dim=-1, keepdim=True)
        mask = n > e
        result = vec.clone()
        result[mask.expand_as(vec)] = vec[mask.expand_as(vec)] * (e / n[mask.expand_as(vec)])
        return result


# ------------------------ 纯算法VNCMD ------------------------
class VNCMD_Complex:
    """
    复数信号的VNCMD算法实现（纯算法版本，无可学习参数）

    基于原始VNCMD论文的实现，支持：
    - 复数信号输入 (batch_size, T)
    - 批量处理
    - 可变模态数（带padding支持）
    """

    def __init__(self, fs, alpha=1e-3, beta=0.1):
        """
        初始化VNCMD算法

        Args:
            fs: 采样频率
            alpha: 数据保真度参数
            beta: 平滑度参数
        """
        self.fs = fs
        self.alpha = alpha
        self.beta = beta

    def __call__(self, s, eIF, var=0.0, tol=1e-7, max_iter=50, mode_mask=None):
        """
        VNCMD分解主函数

        Args:
            s: 输入信号 (batch_size, T) 或 (T,) - 复数
            eIF: 初始瞬时频率 (batch_size, K, T) 或 (K, T) - 实数
            var: 噪声方差
            tol: 收敛容差
            max_iter: 最大迭代次数
            mode_mask: 模态有效性mask (batch_size, K) 或 (K,)

        Returns:
            dict: 包含分解结果的字典
                - 'eIF': 最终瞬时频率
                - 'IA': 瞬时幅度
                - 'reconstructed': 重构信号
                - 'modes': 各模态信号
                - 'iterations': 实际迭代次数
        """
        # 统一处理输入维度
        single_signal = False
        if s.dim() == 1:
            s = s.unsqueeze(0)  # (1, T)
            eIF = eIF.unsqueeze(0)  # (1, K, T)
            if mode_mask is not None:
                mode_mask = mode_mask.unsqueeze(0)  # (1, K)
            single_signal = True

        device = s.device
        batch_size, T = s.shape
        K = eIF.shape[1]  # 模态数

        # 自动检测有效模态
        if mode_mask is None:
            mode_mask = self._detect_active_modes(eIF)

        # 创建二阶差分算子
        e = torch.ones(T, dtype=torch.float32, device=device)
        e2 = -2 * e.clone()
        e2[0] = -1
        e2[-1] = -1
        oper = torch.diag(e[:-1], -1) + torch.diag(e2) + torch.diag(e[:-1], 1)
        opedoub = oper.T @ oper

        # 初始化变量
        results = []
        for b in range(batch_size):
            result = self._vncmd_single(
                s[b], eIF[b], mode_mask[b], opedoub, var, tol, max_iter
            )
            results.append(result)

        # 合并批量结果
        final_result = self._merge_batch_results(results)

        # 如果输入是单个信号，去掉批量维度
        if single_signal:
            for key in ['eIF', 'IA', 'reconstructed', 'modes']:
                if key in final_result:
                    final_result[key] = final_result[key].squeeze(0)

        return final_result

    def _detect_active_modes(self, eIF):
        """检测有效模态"""
        batch_size, K, T = eIF.shape
        mode_mask = torch.zeros((batch_size, K), device=eIF.device)

        for b in range(batch_size):
            for k in range(K):
                if not torch.allclose(eIF[b, k, :], torch.zeros_like(eIF[b, k, :]), atol=1e-6):
                    mode_mask[b, k] = 1.0

        return mode_mask

    def _vncmd_single(self, s, eIF, mode_mask, opedoub, var, tol, max_iter):
        """单个信号的VNCMD分解"""
        device = s.device
        T = s.shape[0]
        K = eIF.shape[0]

        # 初始化
        sinm = torch.zeros((K, T), dtype=torch.float32, device=device)
        cosm = torch.zeros((K, T), dtype=torch.float32, device=device)
        xm = torch.zeros((K, T), dtype=torch.complex64, device=device)
        ym = torch.zeros((K, T), dtype=torch.complex64, device=device)

        # 存储迭代历史
        eIF_history = []
        modes_history = []
        eIF_history.append(eIF.clone())

        # 初始化 xm, ym
        for k in range(K):
            if mode_mask[k] == 0:  # 跳过无效模态
                continue

            phase = 2 * torch.pi * torch.cumsum(eIF[k, :], dim=0) / self.fs
            sinm[k, :] = torch.sin(phase)
            cosm[k, :] = torch.cos(phase)
            xm[k, :] = s.real * cosm[k, :] + s.imag * sinm[k, :]
            ym[k, :] = s.real * sinm[k, :] - s.imag * cosm[k, :]

        # 计算初始累积项
        sum_x = (xm * cosm).sum(dim=0)
        sum_y = (ym * sinm).sum(dim=0)
        lamuda = torch.zeros(T, dtype=torch.complex64, device=device)

        # 初始模态信号
        initial_modes = xm * cosm.to(torch.complex64) + ym * sinm.to(torch.complex64)
        modes_history.append(initial_modes.clone())

        # 迭代优化
        iter_count = 1
        sDif = tol + 1

        while sDif > tol and iter_count <= max_iter:
            # 动态beta调整
            betathr = min(10 ** (iter_count / 36 - 10), self.beta)

            # 投影操作
            u = projec5(s - sum_x - sum_y - lamuda / self.alpha, var)

            old_modes = (xm * cosm.to(torch.complex64) + ym * sinm.to(torch.complex64)).clone()

            # 更新每个模态
            for k in range(K):
                if mode_mask[k] == 0:  # 跳过无效模态
                    continue

                # 减去当前模态的贡献
                sum_x = sum_x - xm[k, :] * cosm[k, :]
                sum_y = sum_y - ym[k, :] * sinm[k, :]

                # 构建求解系统
                Am = torch.diag(cosm[k, :].to(torch.complex64))
                Bm = torch.diag(sinm[k, :].to(torch.complex64))
                A_mat = 2 / self.alpha * opedoub.to(torch.complex64) + Am.conj().T @ Am
                B_mat = 2 / self.alpha * opedoub.to(torch.complex64) + Bm.conj().T @ Bm

                rhs_x = s - sum_x - sum_y - u - lamuda / self.alpha
                rhs_y = rhs_x.clone()

                try:
                    # 求解线性系统
                    xm[k, :] = torch.linalg.solve(A_mat, Am.conj().T @ rhs_x)
                    ym[k, :] = torch.linalg.solve(B_mat, Bm.conj().T @ rhs_y)
                except:
                    # 如果求解失败，使用伪逆
                    try:
                        A_pinv = torch.linalg.pinv(A_mat)
                        B_pinv = torch.linalg.pinv(B_mat)
                        xm[k, :] = A_pinv @ (Am.conj().T @ rhs_x)
                        ym[k, :] = B_pinv @ (Bm.conj().T @ rhs_y)
                    except:
                        print(f"求解失败，跳过模态 {k}")
                        continue

                # 更新瞬时频率
                xbar = Differ5(xm[k, :].real, 1 / self.fs)
                ybar = Differ5(ym[k, :].real, 1 / self.fs)
                den = xm[k, :].real ** 2 + ym[k, :].real ** 2 + 1e-12
                deltaIF = (xm[k, :].real * ybar - ym[k, :].real * xbar) / (2 * torch.pi * den)

                try:
                    # 平滑更新IF
                    I_reg = torch.eye(T, device=device) + 2 / betathr * opedoub
                    deltaIF = torch.linalg.solve(I_reg, deltaIF)
                    eIF[k, :] = eIF[k, :] - 0.5 * deltaIF

                    # 确保IF为正值且在合理范围内
                    eIF[k, :] = torch.clamp(eIF[k, :], min=1.0, max=self.fs / 2 - 1)
                except:
                    # 简单更新
                    eIF[k, :] = eIF[k, :] - 0.1 * deltaIF
                    eIF[k, :] = torch.clamp(eIF[k, :], min=1.0, max=self.fs / 2 - 1)

                # 更新三角函数
                phase = 2 * torch.pi * torch.cumsum(eIF[k, :], dim=0) / self.fs
                sinm[k, :] = torch.sin(phase)
                cosm[k, :] = torch.cos(phase)

                # 重新加入当前模态的贡献
                sum_x = sum_x + xm[k, :] * cosm[k, :]
                sum_y = sum_y + ym[k, :] * sinm[k, :]

            # 更新拉格朗日乘数
            lamuda = lamuda + self.alpha * (u + sum_x + sum_y - s)

            # 计算收敛性
            current_modes = xm * cosm.to(torch.complex64) + ym * sinm.to(torch.complex64)
            sDif = 0
            active_modes = 0

            for k in range(K):
                if mode_mask[k] == 1:
                    mode_diff = torch.norm(current_modes[k, :] - old_modes[k, :])
                    old_norm = torch.norm(old_modes[k, :])
                    sDif += (mode_diff / (old_norm + 1e-12)) ** 2
                    active_modes += 1

            if active_modes > 0:
                sDif = torch.sqrt(sDif / active_modes)

            # 记录历史
            eIF_history.append(eIF.clone())
            modes_history.append(current_modes.clone())

            iter_count += 1

        # 计算最终结果
        IA = torch.sqrt(xm.real ** 2 + ym.real ** 2)
        reconstructed = (xm * cosm.to(torch.complex64) + ym * sinm.to(torch.complex64)).sum(dim=0)
        modes = xm * cosm.to(torch.complex64) + ym * sinm.to(torch.complex64)

        return {
            'eIF': eIF,
            'IA': IA,
            'reconstructed': reconstructed,
            'modes': modes,
            'iterations': iter_count - 1,
            'converged': sDif <= tol,
            'eIF_history': eIF_history,
            'modes_history': modes_history,
            'xm': xm,
            'ym': ym
        }

    def _merge_batch_results(self, results):
        """合并批量结果"""
        merged = {}

        # 合并张量结果
        for key in ['eIF', 'IA', 'reconstructed', 'modes', 'xm', 'ym']:
            if key in results[0]:
                merged[key] = torch.stack([r[key] for r in results], dim=0)

        # 合并标量结果（取平均或最大值）
        merged['iterations'] = max([r['iterations'] for r in results])
        merged['converged'] = all([r['converged'] for r in results])

        return merged




def test_batch_processing():
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")

    fs = 256
    batch_size = 3
    t = torch.arange(0, 1, 1 / fs, dtype=torch.float32)

    # 创建批量信号
    signals = []
    init_ifs = []

    for b in range(batch_size):
        # 每个批次使用不同的参数
        f1 = 30 + b * 10 + 20 * t
        f2 = 80 - b * 5 + 15 * t

        phase1 = 2 * torch.pi * torch.cumsum(f1, dim=0) / fs
        phase2 = 2 * torch.pi * torch.cumsum(f2, dim=0) / fs

        s1 = (0.6 + b * 0.1) * torch.exp(1j * phase1)
        s2 = (0.4 + b * 0.1) * torch.exp(1j * phase2)
        s = s1 + s2

        signals.append(s)
        init_ifs.append(torch.stack([f1 + torch.randn_like(f1) * 2, f2 + torch.randn_like(f2) * 2]))

    signals_batch = torch.stack(signals, dim=0)  # (batch_size, T)
    init_ifs_batch = torch.stack(init_ifs, dim=0)  # (batch_size, K, T)

    print(f"批量信号形状: {signals_batch.shape}")
    print(f"批量IF形状: {init_ifs_batch.shape}")

    # 运行批量VNCMD
    vncmd = VNCMD_Complex(fs, alpha=1e-3, beta=0.1)
    result = vncmd(signals_batch, init_ifs_batch)

    print(f"结果eIF形状: {result['eIF'].shape}")
    print(f"结果IA形状: {result['IA'].shape}")
    print(f"迭代次数: {result['iterations']}")

    # 简单可视化第一个批次
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t, signals_batch[0].imag, 'b-', label='Original Signal', linewidth=2)
    plt.plot(t, result['reconstructed'][0].imag, 'r--', label='Reconstructed Signal', linewidth=2)
    plt.title('Batch 1 - Signal Reconstruction')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, result['eIF'][0, 0, :], 'b-', label='Estimated IF1', linewidth=1.5)
    plt.plot(t, result['eIF'][0, 1, :], 'g-', label='Estimated IF2', linewidth=1.5)
    plt.plot(t,f1,'r--',label='Real1')
    plt.plot(t, f2, 'r--', label='Real2')
    plt.title('Batch 1 - IF Estimation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("纯算法版VNCMD - 复数信号处理")
    print("特点：")
    print("- 无可学习参数，纯算法实现")
    print("- 支持复数信号输入")
    print("- 支持批量处理")
    print("- 完整的收敛检查和历史记录")

    choice = "2"

    if choice == "2":
        test_batch_processing()
