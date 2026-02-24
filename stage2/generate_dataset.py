import os
import argparse
import torch
import numpy as np
from data import noise
from data.data import gen_signal
import json
from scipy.signal import stft
from scipy.fftpack import fftshift
from data import fr
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='./test_dataset', type=str,
                        help="The output directory where the data will be written.")
    parser.add_argument('--overwrite', action='store_true',default=1,
                        help="Overwrite the content of the output directory")

    parser.add_argument("--n_test", default=120, type=int,
                        help="Number of signals")
    parser.add_argument("--signal_dimension", default=256, type=int,
                        help="Dimension of sinusoidal signal")
    parser.add_argument("--minimum_separation", default=0.5, type=float,
                        help="Minimum distance between spikes, normalized by 1/signal_dim")
    parser.add_argument("--max_freq", default=4, type=int,
                        help="Maximum number of frequency, the distribution is uniform between 1 and max_freq")
    parser.add_argument("--distance", default="normal", type=str,
                        help="Distribution type of the inter-frequency distance")
    parser.add_argument("--amplitude", default="uniform", type=str,
                        help="Distribution type of the spike amplitude")
    parser.add_argument("--floor_amplitude", default=0.1, type=float,
                        help="Minimum spike amplitude (only used for the normal_floor distribution)")
    parser.add_argument('--dB', nargs='+', default=['0', '5', '10', '15', '20', '25', '30'],
                        help='additional dB levels')

    parser.add_argument("--numpy_seed", default=105, type=int,
                        help="Numpy seed")
    parser.add_argument("--torch_seed", default=94, type=int,
                        help="Numpy seed")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite to overcome.".format(args.output_dir))
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'data.args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)


    clean_signals, nfreq,sig_details= gen_signal(num_samples=args.n_test, signal_dim=args.signal_dimension, num_freq=args.max_freq,min_sep= 0.5, distance=args.distance,
                                              amplitude=args.amplitude, floor_amplitude=args.floor_amplitude,
                                              variable_num_freq=True)
    signal_dim = args.signal_dimension
    kernel_param_0 = 0.12 / signal_dim
    xgrid = np.linspace(0, 1, signal_dim, endpoint=False)
    masks, s, fr, nfreq,wgt = fr.fre_pre(xgrid, signal_dim, sig_details, nfreq)
    f1, f2, r, theta, amp, sigpos, siglen, f3 = np.split(sig_details,8,axis=1)

    # s, f1,f2, nfreq,r,theta,amp,sp,sl,f3= gen_signal(
    #     num_samples=args.n_test,
    #     signal_dim=args.signal_dimension,
    #     num_freq=args.max_freq,
    #     min_sep=args.minimum_separation,
    #     distance=args.distance,
    #     amplitude=args.amplitude,
    #     floor_amplitude=args.floor_amplitude,
    #     variable_num_freq=True)
    #
    # signal_dim = args.signal_dimension
    # kernel_param_0 = 0.12 / signal_dim
    # xgrid = np.linspace(-0.5, 0.5, signal_dim, endpoint=False)
    # test,wgt = fr.freq2fr(f1, f2, xgrid, param=kernel_param_0, r=r, nfreq=nfreq, sig_dim=signal_dim, theta=theta,amp=amp,sigpos=sp,siglen=sl,f3=f3)
    np.save(os.path.join(args.output_dir, 'infdB'), clean_signals)
    # np.save(os.path.join(args.output_dir, 'f1'), f1)
    # np.save(os.path.join(args.output_dir, 'f2'), f2)
    # np.save(os.path.join(args.output_dir, 'theta'), theta)
    # np.save(os.path.join(args.output_dir, 'r'), r)
    np.save(os.path.join(args.output_dir, 'nfreq'), nfreq)
    # np.save(os.path.join(args.output_dir, 'amp'), amp)
    # np.save(os.path.join(args.output_dir, 'sp'), sigpos)
    # np.save(os.path.join(args.output_dir, 'sl'), siglen)
    # np.save(os.path.join(args.output_dir, 'f3'), f3)
    np.save(os.path.join(args.output_dir, 'sig_details'), sig_details)

    eval_snrs = [float(x) for x in args.dB]

    for k, snr in enumerate(eval_snrs):
        noisy_signals = noise.noise_torch(torch.tensor(clean_signals), snr, 'gaussian').cpu()
        np.save(os.path.join(args.output_dir, '{}dB'.format(float(args.dB[k]))), noisy_signals)

    # data_dir = 'test_dataset'
    # db = ['20.0dB.npy']
    # signal_50dB = np.load(os.path.join(data_dir, db[0]))
    # noisy_signals=signal_50dB
    # win = np.hamming(noisy_signals.shape[2]).astype('float32')
    # # signal_50dB_c = noisy_signals[:, 0] * win[None, :] + 1j * noisy_signals[:, 1] * win[None, :]
    # signal_50dB_c = (noisy_signals[:, 0] + 1j * noisy_signals[:, 1])
    # for idx in range(100):
    #     # _, _, ret = stft(signal_50dB_c[idx], nperseg=8, noverlap=7, nfft=128)
    #     # plt.figure(figsize=(11, 7))
    #     # plt.imshow(np.abs(fftshift(ret, axes=0)))
    #
    #     plt.figure(figsize=(11, 7))
    #     plt.imshow((test[idx]).T)
    #     plt.figure(figsize=(11, 7))
    #     plt.imshow((wgt[idx]).T)
    #     plt.show()