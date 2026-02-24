import torch
import matplotlib.pyplot as plt

# ------------------------ 工具函数 ------------------------
def Differ5(y, delta):
    """五点差分计算导数"""
    L = y.shape[-1]
    ybar = torch.zeros(L, dtype=y.dtype, device=y.device)
    ybar[1:-1] = (y[2:] - y[:-2]) / (2*delta)
    ybar[0] = (y[1] - y[0])/delta
    ybar[-1] = (y[-1] - y[-2])/delta
    return ybar

def projec5(vec, var):
    """投影操作，控制噪声"""
    if var == 0:
        return torch.zeros_like(vec)
    M = vec.numel()
    e = torch.sqrt(torch.tensor(M*var, dtype=torch.float64, device=vec.device))
    n = torch.norm(torch.abs(vec))
    if n > e:
        return vec * (e / n)
    else:
        return vec

# ------------------------ VNCMD 复数版本 ------------------------
def VNCMD_complex(s, fs, eIF, alpha, beta, var, tol=1e-7, max_iter=50):
    device = s.device
    N = s.shape[-1]
    K = eIF.shape[0]
    t = torch.arange(N, dtype=torch.float64, device=device) / fs

    # 二阶差分矩阵
    e = torch.ones(N, dtype=torch.float64, device=device)
    e2 = -2*e
    e2[0] = -1
    e2[-1] = -1
    oper = torch.diag(e[:-1], -1) + torch.diag(e2) + torch.diag(e[:-1], 1)
    opedoub = oper.T @ oper

    # 初始化
    sinm = torch.zeros((K, N), dtype=torch.float64, device=device)
    cosm = torch.zeros((K, N), dtype=torch.float64, device=device)
    xm = torch.zeros((K, N), dtype=torch.complex128, device=device)
    ym = torch.zeros((K, N), dtype=torch.complex128, device=device)
    IFsetiter = torch.zeros((K, N, max_iter+1), dtype=torch.float64, device=device)
    smsetiter = torch.zeros((K, N, max_iter+1), dtype=torch.complex128, device=device)
    IFsetiter[:, :, 0] = eIF

    # 初始化 xm, ym
    for i in range(K):
        phase = 2*torch.pi*torch.cumsum(eIF[i, :], dim=0)/fs
        sinm[i, :] = torch.sin(phase)
        cosm[i, :] = torch.cos(phase)
        xm[i, :] = s.real * cosm[i, :] + s.imag * sinm[i, :]
        ym[i, :] = s.real * sinm[i, :] - s.imag * cosm[i, :]
        smsetiter[i, :, 0] = xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]

    sum_x = (xm * cosm).sum(dim=0)
    sum_y = (ym * sinm).sum(dim=0)
    lamuda = torch.zeros(N, dtype=torch.complex128, device=device)

    iter = 1
    sDif = tol + 1

    while sDif > tol and iter <= max_iter:
        betathr = min(10**(iter/36-10), beta)
        u = projec5(s - sum_x - sum_y - lamuda/alpha, var)

        for i in range(K):
            sum_x = sum_x - xm[i, :] * cosm[i, :]
            sum_y = sum_y - ym[i, :] * sinm[i, :]

            # ------------------- xm/ym 更新（复数矩阵） -------------------
            Am = torch.diag(cosm[i, :].to(torch.complex128))
            Bm = torch.diag(sinm[i, :].to(torch.complex128))
            A_mat = 2/alpha * opedoub.to(torch.complex128) + Am.conj().T @ Am
            B_mat = 2/alpha * opedoub.to(torch.complex128) + Bm.conj().T @ Bm
            rhs_x = s - sum_x - sum_y - u - lamuda/alpha
            rhs_y = rhs_x.clone()
            xm[i, :] = torch.linalg.solve(A_mat, Am.conj().T @ rhs_x)
            ym[i, :] = torch.linalg.solve(B_mat, Bm.conj().T @ rhs_y)

            # ------------------- IF 更新 -------------------
            xbar = Differ5(xm[i, :].real, 1/fs)
            ybar = Differ5(ym[i, :].real, 1/fs)
            den = xm[i, :].real**2 + ym[i, :].real**2 + 1e-12
            deltaIF = (xm[i, :].real*ybar - ym[i, :].real*xbar)/(2*torch.pi*den)
            deltaIF = torch.linalg.solve(torch.eye(N, device=device) + 2/betathr*opedoub, deltaIF)
            eIF[i, :] = eIF[i, :] - 0.5*deltaIF

            # ------------------- 更新 sin/cos -------------------
            phase = 2*torch.pi*torch.cumsum(eIF[i, :], dim=0)/fs
            sinm[i, :] = torch.sin(phase)
            cosm[i, :] = torch.cos(phase)

            sum_x = sum_x + xm[i, :] * cosm[i, :]
            sum_y = sum_y + ym[i, :] * sinm[i, :]
            smsetiter[i, :, iter] = xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]

        IFsetiter[:, :, iter] = eIF
        lamuda = lamuda + alpha * (u + sum_x + sum_y - s)

        # 收敛判断
        sDif = 0
        for i in range(K):
            sDif += ((smsetiter[i, :, iter] - smsetiter[i, :, iter-1]).norm() / (smsetiter[i, :, iter-1].norm()+1e-12))**2

        iter += 1

    IFmset = IFsetiter[:, :, :iter]
    smset = smsetiter[:, :, :iter]
    IA = torch.sqrt(xm.real**2 + ym.real**2)

    return IFmset, IA, smset


fs = 256
t = torch.arange(0, 1, 1/fs, dtype=torch.float64)
f1 = -190 - 80 * t  # mode 1 IF
f2 = 50 + 60 * t  # mode 2 IF
phase1 = 2*torch.pi*torch.cumsum(f1, dim=0)/fs
phase2 = 2*torch.pi*torch.cumsum(f2, dim=0)/fs
s1 = torch.exp(1j*phase1)
s2 = torch.exp(1j*phase2)
s = s1 + s2

# eIF = torch.stack([torch.ones(N) * 150, torch.ones(N) * 50])
eIF = torch.stack([f1 + torch.randn_like(f1) * 2, f2 + torch.randn_like(f2) * 2])
# eIF=torch.stack([f1, f2])

# VNCMD 参数
alpha = 3e-4
beta = 1e-3
var = 0.0
tol = 1e-7

IFmset, IA, smset = VNCMD_complex(s, fs, eIF, alpha, beta, var, tol)

# ----------------- 画图 -----------------
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(t, smset[1, :, -1].real, label='Estimated mode 2', color='green')
plt.plot(t, s2.real, '--', label='True mode 2', color='orange', alpha=0.5)
plt.title('Estimated modes vs True modes')
plt.legend()


plt.subplot(3,1,2)
plt.plot(t, IFmset[0, :, -1], label='Estimated IF mode 1', color='blue')
plt.plot(t, f1, '--', label='True IF mode 1', color='red', alpha=0.5)
plt.plot(t, IFmset[1, :, -1], label='Estimated IF mode 2', color='green')
plt.plot(t, f2, '--', label='True IF mode 2', color='orange', alpha=0.5)
plt.title('Estimated IF vs True IF')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, smset[0, :, -1].real, label='Estimated mode 1', color='blue')
plt.plot(t, s1.real, '--', label='True mode 1', color='red', alpha=0.5)

plt.legend()

plt.tight_layout()
plt.show()
