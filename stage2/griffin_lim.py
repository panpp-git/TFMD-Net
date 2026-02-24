
from complexLayers import custom_stft,custom_istft,ComplexConv2d
import torch.nn.functional as F
from data.loss import complex_cosine_loss
from scipy.signal import hilbert
import torch
import torch.nn as nn
import numpy as np


class PhaseNet(nn.Module):
    """改进版 phase_net，输出 (cos, sin) 避免 wrap-around"""
    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 输出 2 通道：cos(delta), sin(delta)
        self.out = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.encoder(x)
        delta = self.out(feat)  # (B, 2, F, T)

        # 归一化为单位向量
        eps = 1e-8
        norm = torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True)) + eps
        delta_unit = delta / norm  # (B, 2, F, T)

        # 转换为相位残差
        residual_phase = torch.atan2(delta_unit[:, 1, ...], delta_unit[:, 0, ...])  # (B, F, T)
        return residual_phase


class DGRL(nn.Module):
    def __init__(self, n_fft=256, winlen=32, n_iter=500):
        super().__init__()
        self.n_iter = n_iter
        self.winlen = winlen
        self.phase_net = PhaseNet(in_ch=3, base_ch=16)


    def forward(self, mag_spec, phase=None):
        """
        输入:
            mag_spec - 幅度谱 (batch, freq, time)
            phase - 初始相位 (batch, freq, time)，可选
        输出:
            waveform - 重构波形 (batch, time)
        """
        batch, freq, time = mag_spec.shape

        # 如果没给相位，随机初始化
        if phase is None:
            phase = torch.rand(batch, freq, time, device=mag_spec.device) * 2 * np.pi

        for _ in range(self.n_iter):
            # 构建复数谱
            complex_spec = mag_spec * torch.exp(1j * phase)

            # 时域重建
            waveform = custom_istft(complex_spec, freq, time, self.winlen)

            # 重新计算 STFT
            new_spec = custom_stft(waveform, freq, self.winlen)

            # 拼接输入 [mag, real, imag]
            spec_inputs = torch.stack([
                mag_spec, new_spec.real, new_spec.imag
            ], dim=1)  # (batch, 3, freq, time)

            # 预测相位残差
            residual_phase = self.phase_net(spec_inputs) # (batch, freq, time)

            # 更新相位
            phase = torch.angle(new_spec + 1e-6) + residual_phase

        # 裁剪补偿 STFT 窗长偏移
        waveform = waveform[..., (self.winlen - 1) // 2:(self.winlen - 1) // 2 + time]

        return waveform



# if __name__ == "__main__":
#     batch_size = 2
#     time_length = 256  # 1秒音频（16kHz采样率）
#     n_fft = 256
#     hop_length = 1
#     freq_bins = n_fft
#     winlen=32
#
#     t=np.linspace(0,1,256,endpoint=False)
#     # 目标波形（模拟真实数据）
#
#     target_waveform1 = torch.from_numpy(0.5 * np.cos( 2 * torch.pi * (36 * t**2-60*t)))
#     target_waveform2 = torch.from_numpy(0.5 * np.cos(2 * torch.pi * (36 * t ** 2 - 30 * t)))
#     target_waveform=target_waveform1+target_waveform2
#     #+ 0.1 * np.exp(-1j * 2 * torch.pi * 60 * t))
#     noisy_signal = torch.from_numpy(hilbert(target_waveform, axis=-1)).to(torch.complex128)
#     pad = ((winlen-1) // 2, winlen // 2)
#     x_complex = F.pad(noisy_signal, pad)[None,:]
#     spec=custom_stft(x_complex,n_fft,winlen)
#     # mag_spec = torch.abs(spec) / torch.max((torch.abs(spec)))
#     ang = torch.angle(spec)
#
#     x_complex2 = torch.from_numpy(hilbert(target_waveform2, axis=-1)).to(torch.complex128)
#     x_complex2 = F.pad(x_complex2, pad)[None, :]
#     spec1 = custom_stft(x_complex2, n_fft, winlen)
#     mag_spec = torch.abs(spec1) / torch.max((torch.abs(spec1)))
#
#
#     model = DGRL(n_fft=256, winlen=32, n_iter=500)
#     output_waveform = model(mag_spec, ang)
#     # output_waveform=output_waveform[...,(winlen-1) // 2:(winlen-1) // 2+time_length]
#     metric=(F.cosine_similarity(output_waveform.real, target_waveform2[None,:], dim=1))
#
#     # import matplotlib.pyplot as plt
#     # plt.plot(output_waveform.real[0]/torch.max(output_waveform.real[0].abs()),'b')
#     # plt.plot(target_waveform/torch.max(target_waveform.abs()),'r')
#
#     # metric=complex_cosine_loss(output_waveform.real,target_waveform[None,:])
#     # 尺寸检
#     print("输入幅度谱尺寸:", mag_spec.shape)  # 输出: torch.Size([2, 513, 63])
#     print("metric:", metric)  # 输出: torch.Size([2, 16000])
#
#     # 定义损失函数
#     loss_fn = nn.MSELoss()
#     loss = loss_fn(output_waveform, noisy_signal)
#     loss.backward()
#
#     # 检查梯度是否存在
#     print("相位网络梯度是否非空:",
#           model.phase_net[0].weight.grad is not None)  # 输出: True
#           model.phase_net[0].weight.grad is not None)  # 输出: True