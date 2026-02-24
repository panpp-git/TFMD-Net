#!/usr/bin/env python
# coding: utf-8
import h5py
import os
import numpy as np
import torch
import util
import matplotlib.pyplot as plt
from data import fr
from scipy.signal import stft
from scipy.fftpack import fftshift
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F



cResFreq_path = 'checkpoint/TFA-Net_model2/fr/epoch_30.pth'
data_dir = 'test_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util.load(cResFreq_path, 'layer1', 'cpu')
fr_module.cpu()
fr_module.eval()


#load data
sig_details=np.load(os.path.join(data_dir, 'sig_details.npy'))
nfreq=np.load(os.path.join(data_dir, 'nfreq.npy'))
signal_dim=256
kernel_param_0 = 0.12/ signal_dim


db=['20.0dB.npy']
xgrid = np.linspace(0, 1, 256, endpoint=False)
masks, s, fr, nfreq, wgt = fr.fre_pre(xgrid, signal_dim, sig_details, nfreq)

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
            segmask,insmask= fr_module(torch.tensor(signal_50dB[idx][None]))
            #fr_50dB,out


        plt.figure(figsize=(11,7))
        fr_50dB =segmask[0].sigmoid().cpu().data.numpy()
        plt.imshow(((fr_50dB)))


        for i in range(10):
            plt.figure()
            plt.imshow(insmask[0,i].sigmoid().cpu().data.numpy())

        plt.show()



