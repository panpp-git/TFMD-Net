import torch
import math
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
# ------------------ cumtrapz for PyTorch ------------------
# def cumtrapz_torch(y, x):
#     """
#     PyTorch version of MATLAB cumtrapz
#     y: 1D tensor (N,)
#     x: 1D tensor (N,)
#     returns: cumulative trapezoidal integral, same shape as y
#     """
#     y = y.flatten()
#     x = x.flatten()
#     N = y.shape[0]
#     if N < 2:
#         return torch.zeros_like(y)
#     dx = x[1:] - x[:-1]
#     trape = 0.5 * (y[1:] + y[:-1]) * dx
#     cumsum = torch.cat([torch.tensor([0.], dtype=y.dtype, device=y.device),
#                         torch.cumsum(trape, dim=0)])
#     return cumsum
#
# # ------------------ Differ5 (exact MATLAB version) ------------------
# def differ5_torch(y, delta):
#     """
#     MATLAB Differ5 equivalent in PyTorch
#     y: 1D tensor (N,)
#     delta: sampling interval
#     returns: derivative ybar, 1D tensor (N,)
#     """
#     y = y.flatten()
#     N = y.numel()
#     ybar = torch.zeros_like(y)
#
#     if N < 2:
#         return torch.zeros_like(y)
#
#     # central differences for interior points
#     ybar[1:-1] = (y[2:] - y[0:-2]) / (2.0 * delta)
#     # forward/backward differences for boundaries
#     ybar[0] = (y[1] - y[0]) / delta
#     ybar[-1] = (y[-1] - y[-2]) / delta
#
#     return ybar
#
# # ------------------ projec5 (exact MATLAB version) ------------------
# def projec5(vec, var):
#     """
#     MATLAB projec5 equivalent in PyTorch
#     vec: 1D tensor
#     var: scalar variance
#     returns: projected vector u
#     """
#     vec = vec.flatten()
#     M = vec.numel()
#     e = math.sqrt(M * float(var))
#     norm_vec = torch.norm(vec)
#     if norm_vec > e:
#         u = e / norm_vec * vec
#     else:
#         u = vec.clone()
#     return u



def differ5_torch(y, delta):
    L = y.shape[0]
    ybar = torch.zeros(L, dtype=y.dtype, device=y.device)
    ybar[1:-1] = (y[2:] - y[:-2]) / (2*delta)
    ybar[0] = (y[1]-y[0])/delta
    ybar[-1] = (y[-1]-y[-2])/delta
    return ybar

def cumtrapz_torch(y, dx):
    """
    MATLAB cumtrapz 近似实现，长度保持一致
    """
    cumsum = torch.zeros_like(y)
    cumsum[1:] = torch.cumsum((y[:-1]+y[1:])*0.5*dx, dim=0)
    return cumsum

def projec5(vec, var):
    M = vec.numel()
    e = torch.sqrt(torch.tensor(M*var, dtype=vec.dtype, device=vec.device))
    u = vec.clone()
    n = torch.norm(vec)
    if n > e:
        u = vec * (e / n)
    return u


def build_opedoub(N, dtype=torch.float64):
    """构造二阶差分矩阵 opedoub，稀疏 CSC 矩阵"""
    main_diag = -2.0 * torch.ones(N, dtype=dtype)
    upper = torch.ones(N-1, dtype=dtype)
    lower = torch.ones(N-1, dtype=dtype)
    main_diag[0] = -1
    main_diag[-1] = -1
    oper = sp.diags([lower.cpu().numpy(), main_diag.cpu().numpy(), upper.cpu().numpy()],
                    offsets=[-1,0,1], format='csc')
    opedoub = oper.T @ oper
    return opedoub

def VNCMD_sparse_torch(s, fs, eIF, alpha, beta, var, tol, iternum=300):
    """
    VNCMD PyTorch 稀疏加速版本
    s: 1D tensor (N,)
    eIF: 2D tensor (K,N)
    """
    device = s.device
    dtype = s.dtype
    K, N = eIF.shape
    t = torch.arange(N, dtype=dtype, device=device) / fs

    # ------------------ sparse second-difference matrix ------------------
    opedoub = build_opedoub(N, dtype=dtype)

    # ------------------ 初始化 ------------------
    sinm = torch.zeros((K,N), dtype=dtype, device=device)
    cosm = torch.zeros((K,N), dtype=dtype, device=device)
    xm = torch.zeros((K,N), dtype=dtype, device=device)
    ym = torch.zeros((K,N), dtype=dtype, device=device)
    ssetiter = torch.zeros((K,N,iternum+1), dtype=dtype, device=device)
    IFsetiter = torch.zeros((K,N,iternum+1), dtype=dtype, device=device)
    IFsetiter[:,:,0] = eIF.clone()
    lamuda = torch.zeros(N, dtype=dtype, device=device)

    for i in range(K):
        phase = 2*torch.pi*cumtrapz_torch(eIF[i,:], 1/fs)
        sinm[i,:] = torch.sin(phase)
        cosm[i,:] = torch.cos(phase)

        # sparse solve using scipy
        A = (2/alpha)*opedoub + sp.diags(cosm[i,:].cpu().numpy()**2, 0)
        B = (2/alpha)*opedoub + sp.diags(sinm[i,:].cpu().numpy()**2, 0)
        rhs_x = (cosm[i,:]*s).cpu().numpy()
        rhs_y = (sinm[i,:]*s).cpu().numpy()
        xm[i,:] = torch.tensor(spla.spsolve(A, rhs_x), dtype=dtype, device=device)
        ym[i,:] = torch.tensor(spla.spsolve(B, rhs_y), dtype=dtype, device=device)
        ssetiter[i,:,0] = xm[i,:]*cosm[i,:] + ym[i,:]*sinm[i,:]

    sum_x = (xm*cosm).sum(dim=0)
    sum_y = (ym*sinm).sum(dim=0)
    sDif = tol + 1
    iter_idx = 1

    # ------------------ 迭代 ------------------
    while sDif > tol and iter_idx <= iternum:
        betathr = min(10**(iter_idx/36-10), beta)
        u = projec5(s - sum_x - sum_y - lamuda/alpha, var)

        for i in range(K):
            lamuda = torch.zeros_like(lamuda)

            # remove relevant component
            sum_x -= xm[i,:]*cosm[i,:]
            sum_y -= ym[i,:]*sinm[i,:]

            # x-update
            rhs_x = ((s - sum_x - sum_y - u - lamuda/alpha) * cosm[i,:]).cpu().numpy()
            A = (2/alpha)*opedoub + sp.diags(cosm[i,:].cpu().numpy()**2, 0)
            xm[i,:] = torch.tensor(spla.spsolve(A, rhs_x), dtype=dtype, device=device)

            # y-update
            rhs_y = ((s - sum_x - sum_y - u - lamuda/alpha) * sinm[i,:]).cpu().numpy()
            B = (2/alpha)*opedoub + sp.diags(sinm[i,:].cpu().numpy()**2, 0)
            ym[i,:] = torch.tensor(spla.spsolve(B, rhs_y), dtype=dtype, device=device)

            # IF 更新
            xbar = differ5_torch(xm[i,:], 1/fs)
            ybar = differ5_torch(ym[i,:], 1/fs)
            deltaIF = (xm[i,:]*ybar - ym[i,:]*xbar) / (xm[i,:]**2 + ym[i,:]**2) / (2*torch.pi)
            deltaIF = torch.tensor(spla.spsolve(sp.eye(N).tocsc() + (2/betathr)*opedoub, deltaIF.cpu().numpy()),
                                   dtype=dtype, device=device)
            eIF[i,:] = eIF[i,:] - 0.5*deltaIF

            # 更新 sin/cos 和 sums
            phase = 2*torch.pi*cumtrapz_torch(eIF[i,:], 1/fs)
            sinm[i,:] = torch.sin(phase)
            cosm[i,:] = torch.cos(phase)
            sum_x += xm[i,:]*cosm[i,:]
            sum_y += ym[i,:]*sinm[i,:]

            ssetiter[i,:,iter_idx] = xm[i,:]*cosm[i,:] + ym[i,:]*sinm[i,:]

        IFsetiter[:,:,iter_idx] = eIF.clone()
        lamuda = lamuda + alpha*(u + sum_x + sum_y - s)

        # convergence
        sDif = 0
        for i in range(K):
            sDif += (torch.norm(ssetiter[i,:,iter_idx] - ssetiter[i,:,iter_idx-1])
                     / torch.norm(ssetiter[i,:,iter_idx-1]))**2

        iter_idx += 1

    IA = torch.sqrt(xm**2 + ym**2)
    return IFsetiter[:,:,:iter_idx], IA, ssetiter[:,:,:iter_idx]




# ------------------ VNCMD main function ------------------
def VNCMD_torch(s, fs, eIF, alpha, beta, var, tol, iternum=300, device=None, dtype=torch.float64):
    """
    PyTorch implementation of MATLAB VNCMD
    s: 1D array-like (N,) or tensor
    fs: sampling frequency
    eIF: 2D array-like (K,N), each row initial IF of each mode
    alpha, beta, var, tol: scalars
    iternum: max iterations (default 300)
    returns: IFmset (K,N,iter), IA (K,N), smset (K,N,iter)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = torch.as_tensor(s, dtype=dtype, device=device).flatten()
    eIF = torch.as_tensor(eIF, dtype=dtype, device=device)
    K, N = eIF.shape
    t = torch.arange(N, dtype=dtype, device=device) / float(fs)

    # second-difference matrix oper
    e = torch.ones(N, dtype=dtype, device=device)
    e2 = -2.0 * torch.ones(N, dtype=dtype, device=device)
    e2[0] = -1.0; e2[-1] = -1.0
    oper = torch.diag(e2) + torch.diag(e[:-1], -1) + torch.diag(e[:-1], 1)
    opedoub = oper.T @ oper

    # initialize
    sinm = torch.zeros((K, N), dtype=dtype, device=device)
    cosm = torch.zeros((K, N), dtype=dtype, device=device)
    xm = torch.zeros((K, N), dtype=dtype, device=device)
    ym = torch.zeros((K, N), dtype=dtype, device=device)
    IFsetiter = torch.zeros((K, N, iternum+1), dtype=dtype, device=device)
    IFsetiter[:, :, 0] = eIF.clone()
    ssetiter = torch.zeros((K, N, iternum+1), dtype=dtype, device=device)
    lamuda = torch.zeros(N, dtype=dtype, device=device)

    # initialization of modes
    for i in range(K):
        phase = 2.0 * math.pi * cumtrapz_torch(eIF[i, :], t)
        sinm[i, :] = torch.sin(phase)
        cosm[i, :] = torch.cos(phase)

        A = (2.0 / alpha) * opedoub + torch.diag(cosm[i, :]**2)
        B = (2.0 / alpha) * opedoub + torch.diag(sinm[i, :]**2)
        xm[i, :] = torch.linalg.solve(A, (cosm[i, :] * s).unsqueeze(1)).squeeze(1)
        ym[i, :] = torch.linalg.solve(B, (sinm[i, :] * s).unsqueeze(1)).squeeze(1)
        ssetiter[i, :, 0] = xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]

    # iterations
    iter_idx = 1
    sum_x = (xm * cosm).sum(dim=0)
    sum_y = (ym * sinm).sum(dim=0)
    sDif = tol + 1.0

    while (sDif > tol) and (iter_idx <= iternum):
        betathr = 10.0**(iter_idx/36.0 - 10.0)
        if betathr > beta:
            betathr = beta

        u = projec5(s - sum_x - sum_y - lamuda / alpha, var)

        for i in range(K):
            lamuda = torch.zeros_like(lamuda)  # optional disable

            cosv = cosm[i, :]
            sinv = sinm[i, :]

            # x-update
            sum_x = sum_x - xm[i, :] * cosv
            rhs_x = cosv * (s - sum_x - sum_y - u - lamuda / alpha)
            A = (2.0 / alpha) * opedoub + torch.diag(cosv**2)
            xm[i, :] = torch.linalg.solve(A, rhs_x.unsqueeze(1)).squeeze(1)
            interx = xm[i, :] * cosv
            sum_x = sum_x + interx

            # y-update
            sum_y = sum_y - ym[i, :] * sinv
            rhs_y = sinv * (s - sum_x - sum_y - u - lamuda / alpha)
            B = (2.0 / alpha) * opedoub + torch.diag(sinv**2)
            ym[i, :] = torch.linalg.solve(B, rhs_y.unsqueeze(1)).squeeze(1)

            # IF update
            ybar = differ5_torch(ym[i, :], 1.0 / fs)
            xbar = differ5_torch(xm[i, :], 1.0 / fs)
            denom = xm[i, :]**2 + ym[i, :]**2
            eps = 1e-12
            deltaIF = (xm[i, :] * ybar - ym[i, :] * xbar) / (denom + eps) / (2.0 * math.pi)
            S = (2.0 / betathr) * opedoub + torch.eye(N, dtype=dtype, device=device)
            deltaIF_sm = torch.linalg.solve(S, deltaIF.unsqueeze(1)).squeeze(1)
            eIF[i, :] = eIF[i, :] - 0.5 * deltaIF_sm

            # update sin/cos
            phase = 2.0 * math.pi * cumtrapz_torch(eIF[i, :], t)
            sinm[i, :] = torch.sin(phase)
            cosm[i, :] = torch.cos(phase)

            # update sums and store
            sum_x = sum_x - interx + xm[i, :] * cosm[i, :]
            sum_y = sum_y + ym[i, :] * sinm[i, :]
            ssetiter[i, :, iter_idx] = xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]

        IFsetiter[:, :, iter_idx] = eIF.clone()
        lamuda = lamuda + alpha * (u + sum_x + sum_y - s)

        # restart
        if torch.norm(u + sum_x + sum_y - s) > torch.norm(s):
            lamuda = torch.zeros_like(lamuda)
            for i in range(K):
                cosv = cosm[i, :]
                sinv = sinm[i, :]
                A = (2.0 / alpha) * opedoub + torch.diag(cosv**2)
                B = (2.0 / alpha) * opedoub + torch.diag(sinv**2)
                xm[i, :] = torch.linalg.solve(A, (cosv * s).unsqueeze(1)).squeeze(1)
                ym[i, :] = torch.linalg.solve(B, (sinv * s).unsqueeze(1)).squeeze(1)
                ssetiter[i, :, iter_idx] = xm[i, :] * cosv + ym[i, :] * sinv
            sum_x = (xm * cosm).sum(dim=0)
            sum_y = (ym * sinm).sum(dim=0)

        # convergence
        sDif = torch.tensor(0.0, dtype=dtype, device=device)
        for i in range(K):
            denom = torch.norm(ssetiter[i, :, iter_idx - 1]) + 1e-12
            num = torch.norm(ssetiter[i, :, iter_idx] - ssetiter[i, :, iter_idx - 1])
            sDif = sDif + (num / denom)**2

        iter_idx += 1

    IFmset = IFsetiter[:, :, :iter_idx]
    smset = ssetiter[:, :, :iter_idx]
    IA = torch.sqrt(xm**2 + ym**2)

    return IFmset, IA, smset



import matplotlib.pyplot as plt
if __name__ == "__main__":
    fs = 256
    t = torch.arange(0, 1, 1 / fs, dtype=torch.float64)
    N = len(t)

    # 两个非线性调频模式
    f1 = 150 - 80 * t  # mode 1 IF
    f2 = 50 + 60 * t  # mode 2 IF
    s1 = torch.sin(2 * torch.pi * torch.cumsum(f1, dim=0) / fs)
    s2 = torch.sin(2 * torch.pi * torch.cumsum(f2, dim=0) / fs)
    s = s1 + s2  # 观测信号

    # 初始 IF 猜测
    # eIF = torch.stack([torch.ones(N) * 150, torch.ones(N) * 50])
    eIF=torch.stack([f1+ torch.randn_like(f1) * 2, f2+ torch.randn_like(f2) * 2])
    # eIF=torch.stack([f1, f2])

    # VNCMD 参数
    alpha = 3e-4
    beta = 1e-3
    var = 0.0
    tol = 1e-7

    # ------------------ 运行 VNCMD ------------------
    IFmset, IA, smset = VNCMD_sparse_torch(s, fs, eIF, alpha, beta, var, tol)

    # 最后一次迭代
    IF_final = IFmset[:, :, -1]
    sm_final = smset[:, :, -1]

    # ------------------ 绘图 ------------------
    plt.figure(figsize=(12, 8))



    # 2️⃣ IF 对比
    plt.subplot(3, 1, 1)
    plt.plot(t.cpu(), IF_final[0, :].cpu(), label='Estimated IF mode 1', color='blue')
    plt.plot(t.cpu(), IF_final[1, :].cpu(), label='Estimated IF mode 2', color='black')
    plt.plot(t.cpu(), f1.cpu(), '--', label='True IF mode 1', color='red', alpha=0.5)
    plt.plot(t.cpu(), f2.cpu(), '--', label='True IF mode 2', color='red', alpha=0.5)
    plt.title('Estimated IF vs True IF')
    plt.legend()

    # 3️⃣ 模式信号对比
    plt.subplot(3, 1, 2)
    plt.plot(t.cpu(), sm_final[0, :].cpu(), label='Estimated mode 1', color='blue')
    plt.plot(t.cpu(), s1.cpu(), '--', label='True mode 1', color='red', alpha=0.5)
    plt.title('Estimated modes vs True modes')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t.cpu(), sm_final[1, :].cpu(), label='Estimated mode 2', color='blue')
    plt.plot(t.cpu(), s2.cpu(), '--', label='True mode 2', color='red', alpha=0.5)
    plt.title('Estimated modes vs True modes')
    plt.legend()

    plt.tight_layout()
    plt.show()


