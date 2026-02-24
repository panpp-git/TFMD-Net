import torch.nn as nn
import torch
from complexLayers import ComplexConv1d, ComplexConv1d_dilated, ComplexConv2d

from deformattn import MSDeformAttnPixelDecoder
from mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from griffin_lim import DGRL



# from instanceMask import InstanceMask2Former
def set_layer1_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_TFA_Net_part2(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                                    upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                                    kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # b,c,1,1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # b,c,1,1

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding = (kernel_size -1)//2

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # b,1,h,w
        max_out,_ = torch.max(x, dim=1, keepdim=True)  # b,1,h,w
        x_cat = torch.cat([avg_out, max_out], dim=1)  # b,2,h,w
        x = self.conv(x_cat)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class REDNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=8):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        for i in range(num_layers // 2):
            conv_layers.append(nn.Sequential(
                nn.Conv2d(num_features * (2 ** i), num_features * (2 ** (i + 1)), kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=False)))
            conv_layers.append(nn.Sequential(
                nn.Conv2d(num_features * (2 ** (i + 1)), num_features * (2 ** (i + 1)), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(num_features * (2 ** (i + 1)), num_features * (2 ** (i + 1)), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                # nn.Conv2d(num_features * (2 ** (i + 1)), num_features * (2 ** (i + 1)),kernel_size=3, padding=1),
                # nn.ReLU(inplace=False)
                )
                               )
        for i in range(num_layers // 2, 0, -1):
            if i == 0:
                deconv_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(num_features * (2 ** i), num_features * (2 ** i), kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.ConvTranspose2d(num_features * (2 ** i), num_features * (2 ** i), kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    # nn.ConvTranspose2d(num_features * (2 ** i), num_features * (2 ** i), kernel_size=3, padding=1),
                    # nn.ReLU(inplace=False),
                ))
                deconv_layers.append(
                    nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=3, stride=2, padding=1,
                                       output_padding=1))
            else:
                deconv_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(num_features * (2 ** i), num_features * (2 ** i), kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.ConvTranspose2d(num_features * (2 ** i), num_features * (2 ** i),
                                       kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    # nn.ConvTranspose2d(num_features*(2**i), num_features*(2**i), kernel_size=3, padding=1),
                    # nn.ReLU(inplace=False)
                    ))
                deconv_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(num_features * (2 ** i), num_features * (2 ** (i - 1)), kernel_size=3, padding=1,
                                       stride=2, output_padding=1),
                    nn.ReLU(inplace=False)))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        conv_feats = []
        feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                x = x + conv_feat
                conv_feats_idx += 1
                x = self.relu(x)
                feats.append(x)

        x = x + residual
        x = self.relu(x)
        feats.append(x)

        return x, feats[::-1]


class FrequencyRepresentationModule_TFA_Net(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.n_filters = n_filters
        self.inner = inner_dim
        self.n_layers = n_layers

        self.in_layer1 = ComplexConv2d(1, inner_dim * n_filters, kernel_size=(1, 31), padding=(0, 31 // 2),
                                       bias=False)
        self.in_layer2 = ComplexConv2d(1, inner_dim * n_filters, kernel_size=(1, 63), padding=(0, 63 // 2),
                                       bias=False)
        self.in_layer3 = ComplexConv2d(1, inner_dim * n_filters, kernel_size=(1, 95), padding=(0, 95 // 2),
                                       bias=False)

        self.backbone = REDNet30(self.n_layers, num_features=n_filters * 3)
        self.out_layer = nn.Conv2d(n_filters * 3, 1, (1, 1), padding=(0, 0), bias=False)
        self.deformnet = MSDeformAttnPixelDecoder(input_shape=[24, 48, 96, 192])
        self.decoder = MultiScaleMaskedTransformerDecoder(num_queries=10)
        self.cbam = CBAM(channels=n_filters * 3)




    def forward(self, x):
        bsz = x.size(0)
        inp_real = x[:, 0, :].view(bsz, 1, 1, -1, 1)
        inp_imag = x[:, 1, :].view(bsz, 1, 1, -1, 1)
        inp = torch.cat((inp_real, inp_imag), -1)

        x = self.in_layer1(inp)
        x = x.view(bsz, self.n_filters, self.inner, -1, 2)
        maskMouduleInput1 = (x[..., 0] + 1j * x[..., 1]).abs()
        x = self.in_layer2(inp)
        x = x.view(bsz, self.n_filters, self.inner, -1, 2)
        maskMouduleInput2 = (x[..., 0] + 1j * x[..., 1]).abs()
        x = self.in_layer3(inp)
        x = x.view(bsz, self.n_filters, self.inner, -1, 2)
        maskMouduleInput3 = (x[..., 0] + 1j * x[..., 1]).abs()
        maskMouduleInput = torch.cat((maskMouduleInput1, maskMouduleInput2, maskMouduleInput3), dim=1)
        maskMouduleInput = self.cbam(maskMouduleInput)

        segmask, feats = self.backbone(maskMouduleInput)

        mask_features, transformer_encoder_features, multi_scale_features = self.deformnet(feats)
        insmask = self.decoder(multi_scale_features, mask_features, None)

        segmask = self.out_layer(segmask).squeeze(1)


        return segmask, insmask


class FrequencyRepresentationModule_TFA_Net_part2(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.backbone=FrequencyRepresentationModule_TFA_Net(signal_dim=signal_dim, n_filters=n_filters, n_layers=n_layers, inner_dim=inner_dim,
                 kernel_size=kernel_size, upsampling=upsampling, kernel_out=kernel_out)

        self.recon_module = DGRL(n_fft=256, winlen=32, n_iter=7)

    def forward(self, x):
        segmask, insmask=self.backbone(x)
        return segmask,insmask,self.recon_module








