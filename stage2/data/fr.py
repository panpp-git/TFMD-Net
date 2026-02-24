import numpy as np
from sympy import *
import torch
def freq2fr(f1,f2, xgrid, kernel_type='gaussian', param=None, r=None,nfreq=None,sig_dim=None,theta=None,amp=None,sigpos=None,siglen=None,f3=None):
    if kernel_type == 'gaussian':
        return gaussian_kernel_simplify(f1,f2, xgrid, param, r,nfreq,sig_dim,theta,amp,sigpos,siglen,f3)

# def gaussian_kernel_simplify(f1,f2, xgrid, sigma, r,nfreq,sig_dim,theta,amp,sigpos,siglen,f3):
#     t=Symbol('t')
#     t1=np.linspace(0,1,sig_dim,endpoint=False)
#     fr = np.zeros((f1.shape[0], sig_dim,xgrid.shape[0]))
#     wgt_factor = np.ones((f1.shape[0], sig_dim, xgrid.shape[0]))
#
#     abs_max=-1
#     for n in range(fr.shape[0]):
#         tmp_max=np.max(amp[n,:nfreq[n]])
#         if tmp_max>abs_max:
#             abs_max=tmp_max
#     abs_max=20 * np.log10(10*abs_max + 1)
#     for n in range(fr.shape[0]):
#
#         amp[n, :nfreq[n]] = 20 * np.log10(10*amp[n, :nfreq[n]] + 1)
#         for i in range(nfreq[n]):
#             x11 =  r[n,i] * cos(2 * np.pi * (f1[n,i] * t + f2[n,i] * t**2) + theta[n,i])+f3[n,i]*t
#             ph = diff(x11,t)
#             func=lambdify(t,ph,'numpy')
#             ph1=func(t1)
#             # ph1=ph1-np.floor(ph1/sig_dim)*sig_dim
#             # idx1=np.floor(ph1).astype('int')
#             ph1=sig_dim//2+ph1
#             idx1 = np.round(ph1).astype('int')
#
#
#             # fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
#             #     n, i]]+1,256)] = np.maximum(1,fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
#             #     n, i]]+1,256)])
#             # fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
#             #     n, i]]-1,256)] = np.maximum(1,fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
#             #     n, i]]-1,256)])
#
#
#             fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)] =np.maximum(fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)],1)
#             wgt_factor[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)]=np.maximum(wgt_factor[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)],np.power(abs_max/amp[n,i],2))
#
#
#     return fr,wgt_factor

def gaussian_kernel_simplify(f1,f2, xgrid, sigma, r,nfreq,sig_dim,theta,amp,sigpos,siglen,f3):
    # time data
    xgrid=xgrid[:,None]
    t=Symbol('t')
    t1=np.linspace(0,1,sig_dim,endpoint=False)
    fr = np.zeros((f1.shape[0], sig_dim,xgrid.shape[0]))
    wgt_factor = np.ones((f1.shape[0], sig_dim, xgrid.shape[0]))

    masks=np.zeros((f1.shape[0],11,xgrid.shape[0],sig_dim))
    clss=np.zeros((f1.shape[0],11))
    sigs = np.zeros((f1.shape[0], 10,1,sig_dim,2))

    abs_max=-1
    for n in range(fr.shape[0]):
        tmp_max=np.max(amp[n,:nfreq[n]])
        if tmp_max>abs_max:
            abs_max=tmp_max
    abs_max=20 * np.log10(10*abs_max + 1)
    for n in range(fr.shape[0]):
        fr2 = np.zeros((11, xgrid.shape[0], sig_dim))
        cls = np.ones(11)
        cls[nfreq[n]] = 0
        sig = np.zeros((10, 1, sig_dim,2))

        amp[n, :nfreq[n]] = 20 * np.log10(10*amp[n, :nfreq[n]] + 1)
        for i in range(nfreq[n]):
            x11 =  r[n,i] * cos(2 * np.pi * (f1[n,i] * t + f2[n,i] * t**2) + theta[n,i])+f3[n,i]*t
            ph = diff(x11,t)
            func=lambdify(t,ph,'numpy')
            ph1=func(t1)
            # ph1=ph1-np.floor(ph1/sig_dim)*sig_dim
            # idx1=np.floor(ph1).astype('int')
            ph1=sig_dim//2+ph1
            idx1 = np.round(ph1).astype('int')


            fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]+1,256)] = np.maximum(amp[n, i]/6,fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]+1,256)])
            fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]-1,256)] = np.maximum(amp[n, i]/6,fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]-1,256)])


            fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)] =np.maximum(fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)],amp[n,i])
            wgt_factor[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)]=np.maximum(wgt_factor[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)],np.power(abs_max/amp[n,i],2))
            fr2[i, np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], sig_dim), range(sig_dim)[
                                                                           sigpos[n, i]:sigpos[n, i] + siglen[
                                                                               n, i]]] = 1

            xgrid_t = np.zeros((len(xgrid), 1)).astype('complex128')
            xgrid_t[sigpos[n, i]:sigpos[n, i] + siglen[n, i], 0] = xgrid[sigpos[n, i]:sigpos[n, i] + siglen[n, i], 0]
            sig_complex = amp[n, i] * np.exp(2j * np.pi * r[n, i] * np.cos(
                2 * np.pi * (f1[n, i] * xgrid_t.T + f2[n, i] * np.power(xgrid_t.T, 2)) + theta[n, i]) + 2j * np.pi * f3[
                                         n, i] * xgrid_t.T)
            sig[i,:,:,0]=sig_complex.real
            sig[i,:,:,1]=sig_complex.imag


        # bac = np.sum(fr2, axis=0)
        # bac[bac > 1] = 1
        # bac = 1 - bac
        # fr2[nfreq[n], :, :] = bac
        fr2 = fr2.astype('bool')
        cls = cls.astype('int64')
        clss[n] = cls
        masks[n] = fr2
        sigs[n]=sig

    return clss,masks,sigs,fr.transpose((0,2,1)),wgt_factor.transpose((0,2,1))

def fre_pre(xgrid,sig_dim,sig_details,nfreq):
    # time data

    f1, f2, r, theta, amp, sigpos, siglen, f3 = np.split(sig_details,8,axis=1)
    sigpos=sigpos.astype(np.int32)
    siglen=siglen.astype(np.int32)
    xgrid=xgrid[:,None]
    t=Symbol('t')
    t1=np.linspace(0,1,sig_dim,endpoint=False)
    fr = np.zeros((f1.shape[0], sig_dim,xgrid.shape[0]),dtype=np.bool_)
    masks=np.zeros((f1.shape[0],10,xgrid.shape[0],sig_dim),dtype=np.bool_)
    sigs = np.zeros((f1.shape[0], 10,1,sig_dim,2),dtype=np.float32)
    wgt_factor = np.ones((f1.shape[0], sig_dim, xgrid.shape[0]), dtype=np.int32)

    for n in range(fr.shape[0]):
        fr2 = np.zeros((10, xgrid.shape[0], sig_dim),dtype=np.bool_)
        sig = np.zeros((10, 1, sig_dim,2))

        for i in range(nfreq[n]):
            x11=r[n,i] * cos(f1[n,i] * t + theta[n,i])+f3[n,i]*t+f2[n,i]*t**2
            ph = diff(x11,t)
            func=lambdify(t,ph,'numpy')
            ph1=func(t1)


            ph1=sig_dim//2+ph1
            idx1 = np.round(ph1).astype('int')


            # fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
            #     n, i]]+1,256)] = True
            # fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
            #     n, i]]-1,256)] = True
            fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],np.mod(idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],sig_dim)] =True


            # fr2[i, np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]]+1, sig_dim), range(sig_dim)[
            #                                                                sigpos[n, i]:sigpos[n, i] + siglen[
            #                                                                    n, i]]] = True
            # fr2[i, np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]]-1, sig_dim), range(sig_dim)[
            #                                                                sigpos[n, i]:sigpos[n, i] + siglen[
            #                                                                    n, i]]] = True
            fr2[i, np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], sig_dim), range(sig_dim)[
                                                                           sigpos[n, i]:sigpos[n, i] + siglen[
                                                                               n, i]]] = True

            xgrid_t = np.zeros((len(xgrid), 1)).astype('complex128')
            xgrid_t[sigpos[n, i]:sigpos[n, i] + siglen[n, i], 0] = xgrid[sigpos[n, i]:sigpos[n, i] + siglen[n, i], 0]
            sig_complex = amp[n, i] * np.exp(2j * np.pi * r[n, i] * np.cos(
                2 * np.pi * (f1[n, i] * xgrid_t.T + f2[n, i] * np.power(xgrid_t.T, 2)) + theta[n, i]) + 2j * np.pi * f3[
                                         n, i] * xgrid_t.T)
            sig[i,:,:,0]=sig_complex.real
            sig[i,:,:,1]=sig_complex.imag
            wgt_factor[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(
                idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], sig_dim)] += 5
            wgt_factor[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(
                idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]]+1, sig_dim)] += 2
            wgt_factor[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(
                idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]]-1, sig_dim)] += 2
            wgt_factor[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(
                idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]]+2, sig_dim)] += 1
            wgt_factor[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(
                idx1[sigpos[n, i]:sigpos[n, i] + siglen[n, i]]-2, sig_dim)] += 1

        # wgt_factor[n]=process_matrix(wgt_factor[n],11,11)
        #tmp=wgt_factor[n]
        #tmp[tmp>=2]=40
        #wgt_factor[n]=tmp

        masks[n] = fr2
        sigs[n]=sig

    fr = torch.from_numpy(fr)
    masks = torch.from_numpy(np.array(masks))
    sigs = torch.from_numpy(np.array(sigs))
    nfreq=torch.from_numpy(np.array(nfreq)).to(torch.int)
    wgt_factor = torch.from_numpy(wgt_factor)
    # import matplotlib.pyplot as plt
    # plt.imshow(wgt_factor[2].permute(1,0))
    # plt.show()
    return masks,sigs,fr.permute((0,2,1)),nfreq,wgt_factor.permute((0,2,1))








def process_matrix(matrix, k, j):
    """
    处理 m×n 矩阵：若元素值≥2且沿着行方向j范围内存在元素值为2，
    则将该元素及其k×j邻域内所有≥2的元素置为5。

    参数:
        matrix (list[list[int]]): 输入矩阵，元素为1-10。
        k (int): 邻域的行范围。
        j (int): 沿着行方向的列范围。

    返回:
        list[list[int]]: 处理后的矩阵。
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    result = [row.copy() for row in matrix]  # 创建结果矩阵副本

    for i in range(rows):
        for y in range(cols):
            if matrix[i][y] >= 2:
                # 检查行方向j范围内是否存在2（排除自身）
                col_start = max(0, y - j // 2)
                col_end = min(cols, y + j // 2 + 1)
                has_2_in_row = any(
                    matrix[i][z] == 2
                    for z in range(col_start, col_end)
                    if z != y
                )

                # 若存在，则处理邻域k×j
                if has_2_in_row:
                    # 计算邻域范围
                    row_start = max(0, i - k // 2)
                    row_end = min(rows, i + k // 2 + 1)
                    col_start_neighbor = max(0, y - j // 2)
                    col_end_neighbor = min(cols, y + j // 2 + 1)

                    # 修改邻域内所有≥2的元素为5
                    for x in range(row_start, row_end):
                        for z in range(col_start_neighbor, col_end_neighbor):
                            if result[x][z] >= 2:
                                result[x][z] = 5
    return result



