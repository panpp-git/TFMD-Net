import torch
import torch.nn as nn
import numpy as np
from complexLayers import custom_stft,custom_istft,ComplexConv2d
import torch.nn.functional as F
from data.loss import complex_cosine_loss
import matplotlib.pyplot as plt
class DGRL(nn.Module):
    def __init__(self, n_fft=256, winlen=32, n_iter=500,phase=None):
        super().__init__()
        self.n_iter=n_iter
        # 相位修正网络（轻量级）
        # self.phase_net_layer1 = ComplexConv2d(3, 16, kernel_size=3, padding=1)
        # self.phase_net_layer2 = ComplexConv2d(16, 16, kernel_size=3, padding=1)
        # self.phase_net_layer3 = ComplexConv2d(16, 1, kernel_size=3, padding=1)
        # self.ac=torch.nn.Tanh()
        self.phase=phase
        self.winlen=winlen

    def forward(self, mag_spec):
        """
        输入: mag_spec - 幅度谱 (batch, freq, time)
        输出: waveform - 重构波形 (batch, time)
        """
        batch, freq, time = mag_spec.shape
        # 初始化随机相位
        if self.phase is None:
            self.phase = torch.rand(batch, freq, time, device=mag_spec.device) * np.pi*2

        for _ in range(self.n_iter):
            # 构建复数谱
            complex_spec = (mag_spec * torch.exp(1j * self.phase))
            # 时域重建
            waveform=custom_istft(complex_spec, freq, time, self.winlen)

            # 重新计算STFT
            new_spec=custom_stft(waveform,freq,self.winlen)
            # specInps=torch.cat((complex_spec.unsqueeze(1),new_spec.unsqueeze(1)),dim=1)
            # specInps = torch.cat((specInps,(mag_spec * torch.exp(1j * torch.angle(new_spec+1e-6))).unsqueeze(1)),dim=1)
            # specInps=torch.view_as_real(specInps)

            # 相位残差预测
            # residual = self.phase_net_layer1(specInps)
            # residual_amp=(residual[...,0]+1j*(residual[...,1])).abs()
            # residual_phase=torch.angle(residual[...,0]+1j*(residual[...,1])+1e-6)
            # residual=torch.view_as_real(residual_amp*torch.exp(1j*residual_phase))
            # residual = self.phase_net_layer2(residual)
            # residual_amp=(residual[...,0]+1j*(residual[...,1])).abs()
            # residual_phase=torch.angle(residual[...,0]+1j*(residual[...,1])+1e-6)
            # residual = torch.view_as_real(residual_amp * torch.exp(1j * residual_phase))
            # residual = self.phase_net_layer3(residual)
            # residual_phase = torch.angle(residual[..., 0] + 1j * (residual[..., 1]) + 1e-6)

            self.phase = torch.angle(new_spec+1e-6)
        waveform = waveform[..., (self.winlen - 1) // 2:(self.winlen - 1) // 2 + time]
        return waveform

if __name__ == "__main__":
    batch_size = 2
    time_length = 256  # 1秒音频（16kHz采样率）
    n_fft = 256
    hop_length = 1
    freq_bins = n_fft
    winlen=32

    t=np.linspace(0,1,256,endpoint=False)
    # 目标波形（模拟真实数据）

    target_waveform = torch.from_numpy(0.5 * np.exp(1j * 2 * torch.pi * (36 * t**2-60*t))) #+ 0.1 * np.exp(-1j * 2 * torch.pi * 60 * t))
    pad = ((winlen-1) // 2, winlen // 2)
    x_complex = F.pad(target_waveform, pad)[None,:]
    spec=custom_stft(x_complex,n_fft,winlen)
    mag_spec = torch.abs(spec) / torch.max((torch.abs(spec)))
    phase = torch.angle(spec + 1e-6)

    model = DGRL(n_fft=256, winlen=32, n_iter=1000,phase=phase)
    output_waveform = model(mag_spec)
    # output_waveform=output_waveform[...,(winlen-1) // 2:(winlen-1) // 2+time_length]
    metric=complex_cosine_loss(output_waveform,target_waveform[None,:])
    plt.figure()
    plt.plot(output_waveform[0].real/torch.max(output_waveform[0].real.abs()),'k')
    plt.plot(target_waveform.real/torch.max(target_waveform.real.abs()),'r')
    plt.figure()
    plt.plot(output_waveform[0].imag/torch.max(output_waveform[0].imag.abs()),'k')
    plt.plot(target_waveform.imag/torch.max(target_waveform.imag.abs()),'r')

    # 尺寸检查
    print("输入幅度谱尺寸:", mag_spec.shape)  # 输出: torch.Size([2, 513, 63])
    print("metric:", metric)  # 输出: torch.Size([2, 16000])

    # 定义损失函数
    loss_fn = nn.MSELoss()
    loss = loss_fn(output_waveform, target_waveform)
    loss.backward()

    # 检查梯度是否存在
    print("相位网络梯度是否非空:",
          model.phase_net[0].weight.grad is not None)  # 输出: True