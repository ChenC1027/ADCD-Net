import torch
from torch import nn
import torch.nn.functional as F
import lpips as lp
# import ResNet as rs
from collections import namedtuple
from dct_module import DeformableAttention, DeformableAttention2

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)
# def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
#     in_H, in_W = in_tens.shape[2], in_tens.shape[3]
#     return F.interpolate(in_tens, size=out_HW, mode='bilinear', align_corners=False)

def upsample(in_tens, out_HW=(64, 64)):
    upsample_module = nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)
    return upsample_module(in_tens)

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.norm(in_feat, dim=1, keepdim=True)
    return in_feat / (norm_factor + eps)

# def normalize_tensor(in_feat,eps=1e-10):
#     norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
#     return in_feat/(norm_factor+eps)

class Gms(nn.Module):
    def __init__(self, channels):
        super(Gms, self).__init__()
        self.bn0 = nn.BatchNorm2d(channels)

        self.inconv1 = nn.ConvTranspose2d(channels, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.inconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        self.inconv3 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(4)

    def forward(self, x):
        h0 = F.relu(self.bn0(x))
        h1 = self.inconv1(h0)
        h2 = F.relu(self.bn1(h1))

        h3 = self.inconv2(h2)
        h4 = F.relu(self.bn2(h3))

        h5 = self.inconv3(h4)
        h6 = F.relu(self.bn3(h5))

        out = torch.tanh(h6)
        return out


class Gpan(nn.Module):
    def __init__(self, channels):
        super(Gpan, self).__init__()
        self.bn0 = nn.BatchNorm2d(channels)

        self.inconv1 = nn.ConvTranspose2d(channels, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.inconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        self.inconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        h0 = F.relu(self.bn0(x))
        h1 = self.inconv1(h0)
        h2 = F.relu(self.bn1(h1))

        h3 = self.inconv2(h2)
        h4 = F.relu(self.bn2(h3))

        h5 = self.inconv3(h4)
        h6 = F.relu(self.bn3(h5))

        out = torch.tanh(h6)
        return out


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PLMS_Loss(nn.Module):
    def __init__(self, use_dropout=True):
        super(PLMS_Loss, self).__init__()
        self.net = ConNetMS()
        self.chns = [32, 64, 128]
        self.L = len(self.chns)

        # self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        # self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        # self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        # self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        # self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        # self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        # self.lins = nn.ModuleList(self.lins)

    def forward(self, in0_input, in1_input, normalize=False, retPerLayer=False):
        global res
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0_input = 2 * in0_input - 1
            in1_input = 2 * in1_input - 1

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        if retPerLayer:
            return val, res
        else:
            return val


class PLPAN_Loss(nn.Module):
    def __init__(self, use_dropout=True):
        super(PLPAN_Loss, self).__init__()
        self.net = ConNetPAN()
        self.chns = [32, 64, 128]
        self.L = len(self.chns)

    def forward(self, in0_input, in1_input, normalize=False, retPerLayer=False):
        global res
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0_input = 2 * in0_input - 1
            in1_input = 2 * in1_input - 1

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            # feats0[kk], feats1[kk] = lp.normalize_tensor(outs0[kk]), lp.normalize_tensor(outs1[kk])
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        if retPerLayer:
            return val, res
        else:
            return val


class ConNetMS(nn.Module):

    def __init__(self, norm_layer=None):
        super(ConNetMS, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.conv1 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 256, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(4, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.d_ms1 = DeformableAttention(stride=1, distortionmode=True)
        self.d_ms2 = DeformableAttention(stride=1, distortionmode=True)
        self.d_ms3 = DeformableAttention(stride=1, distortionmode=True)


        self.bn1 = norm_layer(self.inplanes)

    def forward(self, x):

        x = self.conv1(x)
        # x = self.d_ms1(x) + x
        h_relu1 = F.relu(x)
        # x = self.bn1(x)

        x = self.conv2(x)
        # x = self.d_ms2(x) + x
        h_relu2 = F.relu(x)

        x = self.conv3(x)
        # x = self.d_ms3(x) + x
        h_relu3 = F.relu(x)
        # x = self.layer4(x)
        # h_relu5 = F.relu(x)

        net_outputs = namedtuple("ConOutputs", ['relu1', 'relu2', 'relu3'])
        out = net_outputs(h_relu1, h_relu2, h_relu3)
        # resnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4'])
        # out = resnet_outputs(h_relu1, h_relu3, h_relu4, h_relu5)

        return out


class ConNetPAN(nn.Module):

    def __init__(self, norm_layer=None):
        super(ConNetPAN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.conv1 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 256, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv4 = nn.Conv2d(4, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.d_pan1 = DeformableAttention2(stride=1, distortionmode=True)
        self.d_pan2 = DeformableAttention2(stride=1, distortionmode=True)
        self.d_pan3 = DeformableAttention2(stride=1, distortionmode=True)
        self.bn1 = norm_layer(self.inplanes)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.d_pan1(x) + x
        h_relu1 = F.relu(x)
        # x = self.bn1(x)

        x = self.conv2(x)
        # x = self.d_pan2(x) + x
        h_relu2 = F.relu(x)

        x = self.conv3(x)
        # x = self.d_pan3(x) + x
        h_relu3 = F.relu(x)
        # x = self.layer4(x)
        # h_relu5 = F.relu(x)

        net_outputs = namedtuple("ConOutputs", ['relu1', 'relu2', 'relu3'])
        out = net_outputs(h_relu1, h_relu2, h_relu3)
        # resnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4'])
        # out = resnet_outputs(h_relu1, h_relu3, h_relu4, h_relu5)

        return out