import numpy as np
import torch
from sympy import Symbol, diff, cos, lambdify

def bresenham_line_clipped(x0, y0, x1, y1, sig_dim):
    """标准 Bresenham，频率越界的点直接丢弃（不 wrap）"""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < sig_dim:
            points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def draw_vertical_at_x(x, y_start, y_end):
    """在固定时间步 x 上，从 y_start 到 y_end（含端点）竖向填充"""
    if y_start <= y_end:
        return [(x, y) for y in range(y_start, y_end + 1)]
    else:
        return [(x, y) for y in range(y_end, y_start + 1)]

def fre_pre(xgrid, sig_dim, sig_details, nfreq):
    """
    生成连续语义 mask (fr)、连续实例 mask (masks) 和连续加权图 (wgt_factor)
    返回:
        masks: (B, 10, F, T)
        sigs:  (B, 10, 1, T, 2)
        fr:    (B, F, T)
        nfreq: (B,)
        wgt:   (B, F, T)
    """
    f1, f2, r, theta, amp, sigpos, siglen, f3 = np.split(sig_details, 8, axis=1)
    sigpos = sigpos.astype(np.int32)
    siglen = siglen.astype(np.int32)
    xgrid = xgrid[:, None]

    t = Symbol('t')
    t1 = np.linspace(0, 1, sig_dim, endpoint=False)

    B = f1.shape[0]
    T = xgrid.shape[0]     # 与你原始代码一致：第三维是时间
    F = sig_dim

    fr = np.zeros((B, F, T), dtype=np.bool_)     # 注意：内部先用 (B, F, T) 操作更直观，最后再 permute 回去
    masks = np.zeros((B, 10, F, T), dtype=np.bool_)
    sigs = np.zeros((B, 10, 1, F, 2), dtype=np.float32)  # 这里沿用你原张量结构（T/F 位置按你实际需要可调整）
    wgt_factor = np.ones((B, F, T), dtype=np.int32)

    for n in range(B):
        fr2 = np.zeros((10, F, T), dtype=np.bool_)
        sig = np.zeros((10, 1, F, 2), dtype=np.float32)

        for i in range(int(nfreq[n])):
            # 相位一阶导（频率轨迹）

            x11 = r[n, i] * cos(f1[n, i] * t + theta[n, i]) + f3[n, i] * t + f2[n, i] * t**2
            ph = diff(x11, t)
            func = lambdify(t, ph, 'numpy')
            ph1 = func(t1)

            # 频率索引（先平移到中心，再四舍五入，再按 F 折叠）
            idx1_raw = np.round(sig_dim // 2 + ph1).astype(int)
            idx1 = np.mod(idx1_raw, F)  # 折叠（wrap），保证在 [0, F-1]

            # 该分量有效的时间范围
            t_start = int(sigpos[n, i])
            t_end   = int(sigpos[n, i] + siglen[n, i])
            time_seq = np.arange(t_start, t_end)
            freq_seq = idx1[t_start:t_end]

            # —— 逐相邻时间步连线 ——
            for k in range(len(time_seq) - 1):
                x0, y0 = int(time_seq[k]),   int(freq_seq[k])
                x1, y1 = int(time_seq[k+1]), int(freq_seq[k+1])

                diff_y = y1 - y0
                # 若在圆周上“长跳”，说明跨边界折叠
                if abs(diff_y) > F // 2:
                    # 折叠情况：不画跨边界的连线，而是在各自时间步竖向到边界
                    if diff_y > 0:
                        # e.g. 5 -> 250 (向上大跳)，真正路径是向下到 0，再从 255 向下到 250
                        pts1 = draw_vertical_at_x(x0, y0, 0)          # x0: y0 -> 0
                        pts2 = draw_vertical_at_x(x1, F-1, y1)        # x1: 255 -> y1
                    else:
                        # e.g. 250 -> 5 (向下大跳)，真正路径是向上到 255，再从 0 向上到 5
                        pts1 = draw_vertical_at_x(x0, y0, F-1)        # x0: y0 -> 255
                        pts2 = draw_vertical_at_x(x1, 0, y1)          # x1: 0  -> y1
                    pts = pts1 + pts2
                else:
                    # 非折叠：标准 Bresenham 连线（不 wrap、不跨界）
                    pts = bresenham_line_clipped(x0, y0, x1, y1, F)

                # 写入 fr / masks / wgt
                for xx, yy in pts:
                    if 0 <= yy < F and 0 <= xx < T:
                        fr[n, yy, xx] = True
                        fr2[i, yy, xx] = True
                        wgt_factor[n, yy, xx] += 5
                        # 邻域权重（频率方向）
                        for dy in (-1, 1):
                            y2 = yy + dy
                            if 0 <= y2 < F:
                                wgt_factor[n, y2, xx] += 2
                        for dy in (-2, 2):
                            y2 = yy + dy
                            if 0 <= y2 < F:
                                wgt_factor[n, y2, xx] += 1

            # —— 复信号（与你原始逻辑一致）——
            xgrid_t = np.zeros((len(xgrid), 1)).astype('complex128')
            xgrid_t[t_start:t_end, 0] = xgrid[t_start:t_end, 0]
            sig_complex = amp[n,i]*np.exp(2j*np.pi*(r[n,i]*np.cos(f1[n,i]*xgrid_t.T+theta[n,i])+f3[n,i]*xgrid_t.T+f2[n,i]*np.power(xgrid_t.T,2.)))
            sig[i, 0, :, 0] = sig_complex.real[0, :F]  # 根据你后续用法，必要时可把 F/T 对齐再改
            sig[i, 0, :, 1] = sig_complex.imag[0, :F]

        masks[n] = fr2
        sigs[n] = sig

    # from scipy.signal import stft
    # import matplotlib.pyplot as plt
    # test_sigs=np.sum(sigs,axis=1).squeeze()
    # for kk in range(test_sigs.shape[0]):
    #     _, _, ret = stft(test_sigs[kk,:,0]+1j*test_sigs[kk,:,1], nperseg=64, noverlap=63, nfft=256)
    #     plt.figure()
    #     plt.imshow(np.fft.fftshift(abs(ret), 0))
    #     plt.figure()
    #     plt.imshow(fr[kk])
    #     plt.figure()
    #     plt.imshow(wgt_factor[kk])
    #     # plt.figure()
    #     # plt.imshow(wgt[kk])
    #     a=0

    fr_t = torch.from_numpy(fr)                       # (B, F, T)
    masks_t = torch.from_numpy(masks)                # (B, 10, F, T)
    sigs_t = torch.from_numpy(sigs)                  # (B, 10, 1, F, 2)
    nfreq_t = torch.from_numpy(np.array(nfreq)).to(torch.int)
    wgt_t = torch.from_numpy(wgt_factor)             # (B, F, T)


    return masks_t, sigs_t, fr_t, nfreq_t, wgt_t
