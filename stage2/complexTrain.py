import os
import sys
import time
import argparse
import logging
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from data import dataset
import complexModules4
import util
from data.noise import noise_torch
import torch
from data.si_snr_loss import cal_si_snr

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(False)
from criterion import SetCriterion
from diff_sinkhorn_matcher import SinkHornMatcher
from data.fr import fre_pre
from dice_entropy_loss import BinaryDiceBCELoss
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from complexLayers import custom_stft,custom_istft
from scipy.signal import stft
# os.environ["CUDA_VISIBLE_DEVICES"]='0'




def si_snr_loss_complex(preds, targets, eps=1e-8):

    pred_unit = preds / (torch.abs(preds) + eps)
    target_unit = targets / (torch.abs(targets) + eps)

    direction_loss = F.mse_loss(pred_unit.real, target_unit.real) + \
                     F.mse_loss(pred_unit.imag, target_unit.imag)

    # ---------------- 总损失 ----------------
    total_loss =  direction_loss
    return total_loss



def hungarian_loss(pred, target, nfreq, criteria, recon,target_sig,noisy_sig):
    with torch.no_grad():
        cost_matrix = []
        directions = []
        for i in range(len(nfreq)):
            Temp = pred[i].flatten(1)
            inputs = (pred[i].sigmoid()).flatten(1)
            # inputs[inputs<0.51]=0
            targets = target[i, :nfreq[i], :, :].flatten(1).to(torch.float32)
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

    losses = 0
    dices = 0
    recon_loss=0
    for i, idx in enumerate(indices):
        dire = directions[i][idx[0], idx[1]]
        for k in range(len(dire)):
            if dire[k] == 1:
                target[i, idx[1][k]] = torch.roll(target[i, idx[1][k]], 1, dims=0)
            elif dire[k] == 2:
                target[i, idx[1][k]] = torch.roll(target[i, idx[1][k]], -1, dims=0)

        loss, dice = criteria(pred[i, idx[0]].unsqueeze(1), target[i, idx[1]])

        losses = loss + losses
        dices = dices + dice

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
        cur_target=(target_sig[i, idx[1], 0, :, 0] + 1j * target_sig[i, idx[1], 0, :, 1]).unsqueeze(1)
        recon_loss=recon_loss+si_snr_loss_complex(recon_sigs.squeeze(1),cur_target.squeeze(1))

        a=1

    return losses / len(indices), recon_loss/len(indices)


def train_frequency_representation(args, fr_module, fr_optimizer, fr_criterion, fr_scheduler, train_loader, val_loader,
                                   xgrid, epoch, tb_writer):
    """
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    loss_train_fr, seg_train,ins_train,recon_train = 0, 0,0,0

    for batch_idx, (clean_signal, masks, s, fr, nfreq, wgt) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal = clean_signal.cuda()
            fr, masks, wgt, s = fr.cuda(), masks.cuda(), wgt.cuda(), s.cuda()

        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        abs_max = torch.max(torch.sqrt((pow(noisy_signal[:, 0, :], 2) + pow(noisy_signal[:, 1, :], 2))))

        for i in range(noisy_signal.size()[0]):
            noisy_signal[i][0] = noisy_signal[i][0] / abs_max
            noisy_signal[i][1] = noisy_signal[i][1] / abs_max

        segmask, insmask, recon_module = fr_module(noisy_signal)
        loss_seg_fr, dice_loss = fr_criterion(segmask.unsqueeze(1), fr.to(torch.int), wgt)
        ins_loss, ins_recon_loss = hungarian_loss(insmask, masks, nfreq, fr_criterion, recon_module,s,noisy_signal)
        loss_fr = ins_recon_loss

        # loss_l2=torch.pow((output_fr - fr ), 2)*wgt
        # loss_fr=torch.sum(loss_l2).to(torch.float32)

        # loss_l1 = criterion(seg, target)
        #
        # for k in list(loss_l1.keys()):
        #     if k in criterion.weight_dict:
        #         loss_l1[k] *= criterion.weight_dict[k]
        #     else:
        #         # remove this loss if not specified in `weight_dict`
        #         loss_l1.pop(k)
        #
        # loss_fr = loss_fr+sum(loss_l1.values())
        #
        # loss_fr=loss_fr+torch.mean(-cal_si_snr(clean_signal[:,0,:].transpose(-1,-2).unsqueeze(-1),time_sig[:,0,:].transpose(-1,-2).unsqueeze(-1))).to(torch.float32)
        # loss_fr=loss_fr+torch.mean(-cal_si_snr(clean_signal[:,1,:].transpose(-1,-2).unsqueeze(-1),time_sig[:,1,:].transpose(-1,-2).unsqueeze(-1))).to(torch.float32)

        fr_optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss_fr.backward()

        fr_optimizer.step()
        loss_train_fr += loss_fr.data.item()
        seg_train = seg_train + loss_seg_fr
        ins_train = ins_train+ins_loss
        recon_train=recon_train+ins_recon_loss

    fr_module.eval()
    loss_val_fr, seg_val, ins_val, recon_val = 0, 0,0,0
    with torch.no_grad():
        for batch_idx, (noisy_signal, clean_signal, masks, s, fr, nfreq, wgt) in enumerate(val_loader):
            if args.use_cuda:
                clean_signal, noisy_signal = clean_signal.cuda(), noisy_signal.cuda()
                fr, masks, wgt, s = fr.cuda(), masks.cuda(), wgt.cuda(), s.cuda()

            abs_max = torch.max(torch.sqrt((pow(noisy_signal[:, 0, :], 2) + pow(noisy_signal[:, 1, :], 2))))
            for i in range(noisy_signal.size()[0]):
                noisy_signal[i][0] = noisy_signal[i][0] / abs_max
                noisy_signal[i][1] = noisy_signal[i][1] / abs_max

            segmask, insmask, recon_module = fr_module(noisy_signal)
            seg_loss_fr, dice_loss = fr_criterion(segmask.unsqueeze(1), fr.to(torch.int), wgt)
            ins_loss, ins_recon_loss = hungarian_loss(insmask, masks, nfreq, fr_criterion, recon_module, s,noisy_signal)
            loss_fr =  ins_recon_loss

            loss_val_fr += loss_fr.data.item()
            seg_val = seg_val + seg_loss_fr
            ins_val=ins_val+ins_loss
            recon_val=recon_val+ins_recon_loss

    loss_train_fr = loss_train_fr / args.n_training
    loss_val_fr = loss_val_fr / args.n_validation
    seg_train /= args.n_training
    seg_val /= args.n_validation
    ins_train /= args.n_training
    ins_val /= args.n_validation
    recon_train /=args.n_training
    recon_val /= args.n_validation

    # tb_writer.add_scalar('fr_l2_training', loss_train_fr, epoch)
    # tb_writer.add_scalar('fr_l2_validation', loss_val_fr, epoch)

    fr_scheduler.step(loss_val_fr)
    logger.info(
        "Epochs: %d / %d, Time: %.1f, FR training L2 loss %.2f, FR validation L2 loss %.2f , train [%.2f %.2f %.2f],  Val [%.2f %.2f %.2f]",
        epoch, args.n_epochs_fr, time.time() - epoch_start_time, loss_train_fr*100, loss_val_fr*100, seg_train*100, ins_train*100, recon_train*100, seg_val*100,ins_val*100, recon_val*100)
    if epoch == 10:
        x = 1
    # 返回验证损失用于早停机制
    return loss_val_fr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/TFA-Net_model2', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=256, help='dimensionof the input signal')
    parser.add_argument('--fr_size', type=int, default=256, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=6,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    parser.add_argument('--min_sep', type=float, default=0.5,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='normal_floor', help='spike amplitude distribution')
    parser.add_argument('--floor_amplitude', type=float, default=0.1, help='minimum amplitude of spikes')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=-5, help='snr parameter')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=8, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=8, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=3, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=256, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=1,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')

    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=1000,
                        help='slope of the triangle kernel normalized by signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.12,
                        help='std of the gaussian kernel normalized by signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=5000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=200, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.0001,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--n_epochs_fr', type=int, default=410, help='number of epochs used to train the fr module')
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
        args.device = 'cuda'
    else:
        args.use_cuda = False
        args.device = 'cpu'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    # tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    train_loader = dataset.make_train_data(args)
    val_loader = dataset.make_eval_data(args)

    fr_criterion = BinaryDiceBCELoss()

    pre_path = 'epoch_100.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fr_module, fr_optimizer, fr_scheduler, _, _ = util.load_freeze(pre_path, 'layer1', device)
    fr_module.eval()

    # fr_module = complexModules4.set_layer1_module(args)
    # fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'layer1')


    start_epoch = 1

    logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
            util.model_parameters(fr_module) / 1e6))

    xgrid = np.linspace(0, 1, args.fr_size, endpoint=False)
    # 早停机制参数
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15  # 15个epoch没有改善就停止

    for epoch in range(start_epoch, args.n_epochs_fr + 1):

        if epoch < args.n_epochs_fr:
            val_loss =train_frequency_representation(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer,
                                           fr_criterion=fr_criterion,
                                           fr_scheduler=fr_scheduler, train_loader=train_loader, val_loader=val_loader,
                                           xgrid=xgrid, epoch=epoch, tb_writer=None)
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, f"{args.fr_module_type}_best")
                logger.info(f"New best validation loss: {val_loss:.4f} at epoch {epoch}")
            else:
                patience_counter += 1

            # 早停条件
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} due to no improvement for {early_stop_patience} epochs")
                break

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_fr:
            util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)

