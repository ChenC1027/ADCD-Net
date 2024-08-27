import torch
import torch.nn as nn
import torch.nn.functional as F
from dct_module import DeformableAttention, DeformableAttention2
from g_module import Gms, PLMS_Loss, Gpan, PLPAN_Loss
import tf
import mobilenetV3 as Mb
from DTM import Mamba

class MSNet(nn.Module):
    def __init__(self, channel_hsi, Ms_stout):
        super(MSNet, self).__init__()

        self.Ms_stout = Ms_stout
        self.conv1 = nn.Conv2d(channel_hsi, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.mb_conv2 = Mb.ResidualBlock(128, 256, 3, 2, use_se=False)
        self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1)
        self.mb_conv3 = Mb.ResidualBlock(256, 128, 3, 1, use_se=False)
        self.bn3 = nn.BatchNorm2d(128)

        # self.mb_conv4 = Mb.ResidualBlock(128, 64, 3, 1, use_se=False)
        # self.bn4 = nn.BatchNorm2d(64)

        self.Ms_st1 = nn.Sequential(nn.Linear(16 * 16 * 128, self.Ms_stout), LayerNorm(self.Ms_stout))
        self.Ms_st2 = nn.Sequential(nn.Linear(8 * 8 * 256, self.Ms_stout), LayerNorm(self.Ms_stout))
        self.Ms_st3 = nn.Sequential(nn.Linear(8 * 8 * 128, self.Ms_stout), LayerNorm(self.Ms_stout))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.mb_conv1(x)
        x1 = x.contiguous().view(x.size(0), -1)
        x1 = F.relu(self.Ms_st1(x1)).unsqueeze(1)

        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.mb_conv2(x)
        x2 = x.contiguous().view(x.size(0), -1)
        x2 = F.relu(self.Ms_st2(x2)).unsqueeze(1)

        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mb_conv3(x)
        x3 = x.contiguous().view(x.size(0), -1)
        x3 = F.relu(self.Ms_st3(x3)).unsqueeze(1)

        # x = F.relu(self.bn4(self.mb_conv4(x)))

        return x, x1, x2, x3
        # return x


class PANNet(nn.Module):
    def __init__(self, channel_msi, Pan_stout=2048):
        super(PANNet, self).__init__()

        self.Pan_stout = Pan_stout
        self.conv1 = nn.Conv2d(channel_msi, 128, 3, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.mb_conv2 = Mb.ResidualBlock(128, 256, 3, 2, use_se=False)
        self.bn2 = nn.BatchNorm2d(256)

        # self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(256, 128, 3, stride=2, padding=1)
        self.mb_conv3 = Mb.ResidualBlock(256, 128, 3, 2, use_se=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.mb_conv4 = Mb.ResidualBlock(128, 64, 3, 1, use_se=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.Pan_st1 = nn.Sequential(nn.Linear(32 * 32 * 128, self.Pan_stout), LayerNorm(self.Pan_stout))
        self.Pan_st2 = nn.Sequential(nn.Linear(16 * 16 * 256, self.Pan_stout), LayerNorm(self.Pan_stout))
        self.Pan_st3 = nn.Sequential(nn.Linear(8 * 8 * 128, self.Pan_stout), LayerNorm(self.Pan_stout))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.contiguous().view(x.size(0), -1)
        x1 = F.relu(self.Pan_st1(x1)).unsqueeze(1)

        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.mb_conv2(x)
        x2 = x.contiguous().view(x.size(0), -1)
        x2 = F.relu(self.Pan_st2(x2)).unsqueeze(1)

        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mb_conv3(x)
        x3 = x.contiguous().view(x.size(0), -1)
        x3 = F.relu(self.Pan_st3(x3)).unsqueeze(1)

        # x = F.relu(self.bn4(self.mb_conv4(x)))

        return x, x1, x2, x3

        # return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Dropout(nn.Module):
    def __init__(self):
        super(Dropout, self).__init__()

    def forward(self, x):
        out = F.dropout(x, p=0.2, training=self.training)
        return out


class Net(nn.Module):
    def __init__(self, channel_ms, channel_pan, class_num, Ms_stout=256, Pan_stin=512):
        super(Net, self).__init__()

        self.hide_line = 64
        self.featnet1 = MSNet(channel_ms, Ms_stout)
        self.featnet2 = PANNet(channel_pan, Pan_stin)
        # self.cam = CAM()
        # self.fc11 = nn.Linear(1 * 1 * 1024, 64)
        self.fc2 = nn.Linear(self.hide_line * 1, class_num)

        self.dropout = nn.Dropout()
        self.dct = DeformableAttention(stride=1, distortionmode=True)
        self.dct1 = DeformableAttention2(stride=1, distortionmode=True)
        # self.ch_dct = DeformableCh_Attention(1, 128, distortionmode=True)
        self.gms = Gms(128)
        self.gpan = Gpan(128)
        # self.loss_fn = lp.LPIPS(net='alex')
        self.loss_ms = PLMS_Loss(use_dropout=True)
        self.loss_pan = PLPAN_Loss(use_dropout=True)
        # self.gcn = GcnNet(kernel_size=2, clust_num=9, hide_num=64)
        self.proj_normms = LayerNorm(Ms_stout)
        self.proj_normpan = LayerNorm(Pan_stin)
        self.Ms_st4 = nn.Sequential(nn.Linear(8 * 8 * 128, Ms_stout), LayerNorm(Ms_stout))
        self.Pan_st4 = nn.Sequential(nn.Linear(8 * 8 * 128, Pan_stin), LayerNorm(Pan_stin))
        self.fc1 = nn.Linear(8 * 8 * 128, self.hide_line)
        # self.Ms_sfatt = tf.selfAttention(2, Ms_stout, self.hide_line)
        # self.Pan_sfatt = tf.selfAttention(2, Pan_stin, self.hide_line)
        self.Ms_mb = Mamba(d_model=Ms_stout, out_ch=self.hide_line, expand=1)
        self.Pan_mb = Mamba(d_model=Pan_stin, out_ch=self.hide_line, expand=1)

        self.out_proj_norm = LayerNorm(self.hide_line)

    def forward(self, x, y):
        # Pre-process Image Feature

        Ms, ms_dpl1, ms_dpl2, ms_dpl3 = self.featnet1(x)
        Ms = self.dct(Ms)
        Ms_dpl = self.Ms_mb(self.proj_normms(torch.cat((ms_dpl1, ms_dpl2, ms_dpl3,
                                                F.relu(self.Ms_st4(Ms.contiguous().view(x.size(0), -1))).unsqueeze(1)), dim=1)))
        loss2 = torch.mean(self.loss_ms.forward(self.gms(Ms), x, normalize=True))

        Pan, pan_dpl1, pan_dpl2, pan_dpl3 = self.featnet2(y)
        Pan = self.dct1(Pan)
        Pan_dpl = self.Pan_mb(self.proj_normpan(torch.cat((pan_dpl1, pan_dpl2, pan_dpl3,
                                                   F.relu(self.Pan_st4(Pan.contiguous().view(x.size(0), -1))).unsqueeze(1)), dim=1)))
        loss3 = torch.mean(self.loss_pan.forward(self.gpan(Pan), y, normalize=True))

        x1 = Ms.contiguous().view(x.size(0), -1)
        y1 = Pan.contiguous().view(y.size(0), -1)
        x = x1 + y1

        x = F.relu(self.out_proj_norm(self.fc1(x)))
        Pan_x = F.relu(self.out_proj_norm(torch.mean(Pan_dpl, dim=1).squeeze(1)))
        MS_x = F.relu(self.out_proj_norm(torch.mean(Ms_dpl, dim=1).squeeze(1)))
        x = self.dropout(x)
        x = self.fc2(x)
        Pan_x = self.fc2(Pan_x)
        MS_x = self.fc2(MS_x)


        return x, loss2, loss3, Pan_x, MS_x
        # return x, Pan_x, MS_x
        # return x


class SS_Net(nn.Module):
    def __init__(self, channel_ms, channel_pan, Ms_stout=256, Pan_stin=512):
        super(SS_Net, self).__init__()

        self.hide_line = 64
        self.featnet1 = MSNet(channel_ms, Ms_stout)
        self.featnet2 = PANNet(channel_pan, Pan_stin)
        self.dct = DeformableAttention(stride=1, distortionmode=True)
        self.dct1 = DeformableAttention2(stride=1, distortionmode=True)
        self.loss_pan = PLPAN_Loss(use_dropout=True)
        self.Ms_st4 = nn.Sequential(nn.Linear(8 * 8 * 128, Ms_stout), LayerNorm(Ms_stout))
        self.Pan_st4 = nn.Sequential(nn.Linear(8 * 8 * 128, Pan_stin), LayerNorm(Pan_stin))

    def forward(self, x, y):
        # Pre-process Image Feature
        Ms, _, _, _ = self.featnet1(x)
        Ms = self.dct(Ms)
        Ms_token = F.relu(self.Ms_st4(Ms.contiguous().view(x.size(0), -1))).unsqueeze(1)
        Pan, _, _, _ = self.featnet2(y)
        Pan = self.dct1(Pan)
        Pan_token = F.relu(self.Pan_st4(Pan.contiguous().view(x.size(0), -1))).unsqueeze(1)
        cross_loss = torch.mean(self.loss_pan.forward(Pan, Ms, normalize=True))

        return cross_loss, Ms_token, Pan_token, Ms, Pan


class S_Net(nn.Module):
    def __init__(self, channel_ms, channel_pan, class_num, Ms_stout=256, Pan_stin=512, pool='cls', cls_token='behead'):
        super(S_Net, self).__init__()

        self.hide_line = 64
        self.pool = pool
        self.featnet1 = MSNet(channel_ms, Ms_stout)
        self.featnet2 = PANNet(channel_pan, Pan_stin)
        self.cls_token = cls_token
        self.fc2 = nn.Linear(self.hide_line * 1, class_num)

        # self.outche = nn.Sequential(nn.Conv2d(128 * 2, 128, 3, padding=1), nn.BatchNorm2d(128))

        self.dropout = nn.Dropout()
        self.dct = DeformableAttention(stride=1, distortionmode=True)
        self.dct1 = DeformableAttention2(stride=1, distortionmode=True)
        self.proj_normms = LayerNorm(Ms_stout)
        self.proj_normpan = LayerNorm(Pan_stin)
        self.fc1 = nn.Linear(8 * 8 * 128, self.hide_line)
        self.Ms_mb = Mamba(d_model=Ms_stout, out_ch=self.hide_line, expand=1)
        self.Pan_mb = Mamba(d_model=Pan_stin, out_ch=self.hide_line, expand=1)


        self.out_proj_norm = LayerNorm(self.hide_line)

    # def forward(self, x, y, Ms_token, Pan_token, Ms, Pan):
    def forward(self, x, y, Ms_token, Pan_token, ms_resdual, pan_resdual):
        # Pre-process Image Feature
        token_position = None
        # B = x.size(0)
        Ms, ms_dpl1, ms_dpl2, ms_dpl3 = self.featnet1(x)
        Ms = self.dct(Ms)
        Ms_token = nn.Parameter(Ms_token)
        Ms_dpl = self.Ms_mb(self.proj_normms(torch.cat((ms_dpl1, ms_dpl2, ms_dpl3,
                                                        F.relu(self.Ms_st4(Ms.contiguous().view(x.size(0), -1))).unsqueeze(1), Ms_token), dim=1)))

        Ms = F.relu(nn.Parameter(torch.sigmoid(ms_resdual)) * Ms + Ms)

        Pan, pan_dpl1, pan_dpl2, pan_dpl3 = self.featnet2(y)
        Pan = self.dct1(Pan)
        Pan_token = nn.Parameter(Pan_token)
        if self.cls_token == 'head':
            token_position = 0
        elif self.cls_token == 'behead':
            token_position = 4

        Pan_dpl = self.Pan_mb(self.proj_normpan(torch.cat((pan_dpl1, pan_dpl2, pan_dpl3,
                                                           F.relu(self.Pan_st4(Pan.contiguous().view(x.size(0), -1))).unsqueeze(1), Pan_token), dim=1)))
        Pan = F.relu(nn.Parameter(torch.sigmoid(pan_resdual)) * Pan + Pan)

        x1 = Ms.contiguous().view(x.size(0), -1)
        y1 = Pan.contiguous().view(y.size(0), -1)
        x = x1 + y1

        x = F.relu(self.out_proj_norm(self.fc1(x)))
        Pan_x = F.relu(self.out_proj_norm(torch.mean(Pan_dpl, dim=1).squeeze(1))) if self.pool == 'mean' else F.relu(self.out_proj_norm(Pan_dpl[:, token_position, :].squeeze(1)))
        MS_x = F.relu(self.out_proj_norm(torch.mean(Ms_dpl, dim=1).squeeze(1))) if self.pool == 'mean' else F.relu(self.out_proj_norm(Ms_dpl[:, token_position, :].squeeze(1)))
        x = self.dropout(x)
        x = self.fc2(x)
        Pan_x = self.fc2(Pan_x)
        MS_x = self.fc2(MS_x)

        return x, Pan_x, MS_x


if __name__ == "__main__":
    # torch.randn:用来生成随机数字的tensor，这些随机数字满足标准正态分布（0~1），batch_size=2， 1通道（灰度图像为1），图片尺寸：64x64
    pan = torch.randn(2, 2, 16, 16)
    ms = torch.randn(2, 64, 16, 16)
    # mshpan = torch.randn(2, 1, 64, 64)
    Net = Net(64, 2, 7)
    # out_result = grf_net(ms, pan, mshpan)
    out_result = Net(ms, pan)
    print(out_result)
    print(out_result.shape)

