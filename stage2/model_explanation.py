#!/usr/bin/env python
# coding: utf-8
import h5py
import os
import numpy as np
import torch
import util
import matplotlib.pyplot as plt
from data.fr_corr import fre_pre
from scipy.signal import stft
from scipy.fftpack import fftshift
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from complexLayers import custom_stft,custom_istft
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def hungarian_loss(pred, target, nfreq, recon,noisy_sig):
    with torch.no_grad():
        cost_matrix = []
        directions = []
        for i in range(pred.size()[0]):
            Temp = pred[i].flatten(1)
            inputs = (pred[i].sigmoid()).flatten(1)

            targets = target[i, :nfreq, :, :].flatten(1).to(torch.float32)
            numerator = 2 * torch.einsum("nk,mk->nm", inputs, targets)
            denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
            cost = 1 - (numerator + 1) / (denominator + 1)

            inputs_rotup = torch.roll((pred[i].sigmoid()), shifts=-1, dims=1).flatten(1)
            # inputs_rotup[inputs_rotup < 0.51] =0
            numerator = 2 * torch.einsum("nk,mk->nm", inputs_rotup, targets)
            denominator = inputs_rotup.sum(-1)[:, None] + targets.sum(-1)[None, :]
            cost_up = 1 - (numerator + 1) / (denominator + 1)

            inputs_rotdown = torch.roll((pred[i].sigmoid()), shifts=1, dims=1).flatten(1)
            # inputs_rotdown[inputs_rotdown < 0.51] = 0
            numerator = 2 * torch.einsum("nk,mk->nm", inputs_rotdown, targets)
            denominator = inputs_rotdown.sum(-1)[:, None] + targets.sum(-1)[None, :]
            cost_down = 1 - (numerator + 1) / (denominator + 1)
            cost = torch.concat([cost.unsqueeze(0), cost_up.unsqueeze(0), cost_down.unsqueeze(0)], dim=0)

            inputs_up = torch.roll(pred[i], shifts=-1, dims=1).flatten(1)
            pos = F.binary_cross_entropy_with_logits(
                inputs_up, torch.ones_like(inputs_up), reduction="none"
            )
            neg = F.binary_cross_entropy_with_logits(
                inputs_up, torch.zeros_like(inputs_up), reduction="none"
            )
            loss_up = (torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
                "nc,mc->nm", neg, (1 - targets))) / targets.size(-1)

            inputs_down = torch.roll(pred[i], shifts=1, dims=1).flatten(1)
            pos = F.binary_cross_entropy_with_logits(
                inputs_down, torch.ones_like(inputs_down), reduction="none"
            )
            neg = F.binary_cross_entropy_with_logits(
                inputs_down, torch.zeros_like(inputs_down), reduction="none"
            )
            loss_down = (torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
                "nc,mc->nm", neg, (1 - targets))) / targets.size(-1)

            inputs = Temp
            pos = F.binary_cross_entropy_with_logits(
                inputs, torch.ones_like(inputs), reduction="none"
            )
            neg = F.binary_cross_entropy_with_logits(
                inputs, torch.zeros_like(inputs), reduction="none"
            )
            loss = (torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
                "nc,mc->nm", neg, (1 - targets))) / targets.size(-1)
            loss = torch.concat([loss.unsqueeze(0), loss_up.unsqueeze(0), loss_down.unsqueeze(0)], dim=0)

            total_loss = torch.min(cost + loss, dim=0, keepdim=False).values
            direction = torch.min(cost + loss, dim=0, keepdim=False).indices
            cost_matrix.append(total_loss)
            directions.append(direction)
        indices = [linear_sum_assignment(cost.cpu().detach().numpy()) for cost in cost_matrix]


    for i, idx in enumerate(indices):

        # Recon
        winlen=32
        freq_num = pred.shape[2]
        ori_sig=noisy_sig[i, 0, :] + 1j * noisy_sig[i, 1, :]
        pad = ((winlen - 1) // 2, winlen // 2)
        x_complex = F.pad(ori_sig, pad)[None,:]
        spec = custom_stft(x_complex, freq_num, winlen)

        # import matplotlib.pyplot as plt
        # _, _, ret = stft(x_complex[0].detach().cpu().numpy(), nperseg=64, noverlap=63, nfft=256)
        # plt.figure()
        # plt.imshow(np.fft.fftshift(abs(ret), 0))
        # plt.figure()
        # plt.imshow(spec[0].abs().cpu().numpy())
        # a=1

        ang = torch.angle(spec)
        cur_pred=pred[i, idx[0]].sigmoid()
        recon_sigs=recon(cur_pred, ang).unsqueeze(1)

    return recon_sigs,indices


cResFreq_path = 'checkpoint/TFA-Net_model2/fr_best/epoch_53_best.pth'
data_dir = 'test_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util.load(cResFreq_path, 'layer1', 'cpu')
fr_module.cpu()
fr_module.eval()

query_embed = fr_module.decoder.query_feat.weight.detach().cpu()

print("Query embedding shape:", query_embed.shape)

# -------------------------------
# 4. t-SNE / PCA 可视化
# -------------------------------
# t-SNE
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
q_tsne = tsne.fit_transform(query_embed.numpy())

plt.figure(figsize=(6,6))
plt.scatter(q_tsne[:,0], q_tsne[:,1], c='blue')
plt.title("Mask2Former Query Embedding (t-SNE)")
plt.show()

# PCA
pca = PCA(n_components=2)
q_pca = pca.fit_transform(query_embed.numpy())

plt.figure(figsize=(6,6))
plt.scatter(q_pca[:,0], q_pca[:,1], c='red')
plt.title("Mask2Former Query Embedding (PCA)")
plt.show()


#load data
sig_details=np.load(os.path.join(data_dir, 'sig_details.npy'))
nfreq=np.load(os.path.join(data_dir, 'nfreq.npy'))
signal_dim=256
kernel_param_0 = 0.12/ signal_dim


db=['20.0dB.npy']
xgrid = np.linspace(0, 1, 256, endpoint=False)
masks, s, fr, nfreq, wgt = fre_pre(xgrid, signal_dim, sig_details, nfreq)

for db_iter in range(0,len(db)):
    signal_50dB = np.load(os.path.join(data_dir, db[db_iter]))
    win = np.hamming(signal_50dB.shape[2]).astype('float32')
    signal_50dB_c = signal_50dB[:, 0]*win[None,:] + 1j * signal_50dB[:, 1]*win[None,:]
    signal_50dB_m = signal_50dB[:, 0]  + 1j * signal_50dB[:, 1]

    for idx in range(8,nfreq.shape[0]):
        plt.figure(figsize=(11, 7))
        plt.imshow((fr[idx]))
        print(db_iter,idx)
        with torch.no_grad():

            mv = np.max(np.sqrt(pow(signal_50dB[idx][0], 2) + pow(signal_50dB[idx][1], 2)))
            signal_50dB[idx][0]=signal_50dB[idx][0]/mv
            signal_50dB[idx][1] = signal_50dB[idx][1] / mv

            _,_,ret=stft(signal_50dB_c[idx],nperseg = 32,noverlap=31,nfft = 256)
            segmask,insmask,recon= fr_module(torch.tensor(signal_50dB[idx][None]))
            #fr_50dB,out


        plt.figure(figsize=(11,7))
        fr_50dB =segmask[0].sigmoid().cpu().data.numpy()
        plt.imshow(((fr_50dB)))


        # for i in range(10):
        #     plt.figure()
        #     plt.imshow(insmask[0,i].sigmoid().cpu().data.numpy())
        #
        # plt.show()
        # plt.show()

        cur_mask=masks[idx].unsqueeze(0)
        cur_nfreq=nfreq[idx]
        recon_sig,indices = hungarian_loss(insmask, cur_mask, cur_nfreq, recon, torch.tensor(signal_50dB[idx][None]))

        target_sig=s[idx].unsqueeze(0)
        recon_sig=recon_sig.squeeze(1)
        for i, idx in enumerate(indices):
            target_sigs=(target_sig[i, idx[1], 0, :, 0] + 1j * target_sig[i, idx[1], 0, :, 1])
            target_sigs=target_sigs.detach().numpy()
            recon_sig=recon_sig.detach().numpy()

            target_real=target_sigs.real
            target_imag=target_sigs.imag
            target_real=target_real/np.max(target_real,axis=1)[:,None]
            target_imag=target_imag/np.max(target_imag,axis=1)[:,None]

            recon_real=recon_sig.real
            recon_imag=recon_sig.imag
            recon_real=recon_real/np.max(recon_real,axis=1)[:,None]
            recon_imag=recon_imag/np.max(recon_imag,axis=1)[:,None]

            for j in range(recon_sig.shape[0]):
                plt.figure()
                plt.plot(target_real[j,:],'r')
                plt.plot(recon_real[j,:],'b')
                plt.figure()
                plt.plot(target_imag[j,:],'r')
                plt.plot(recon_imag[j,:],'b')
        plt.show()
        plt.show()








