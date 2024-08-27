import torch
from torch import nn
import torch.nn.functional as F

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        新增modulation 参数： 是DCNv2中引入的调制标量
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 输出通道是2N
        nn.init.constant_(self.p_conv.weight, 0)  # 权重初始化为0
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:  # 如果需要进行调制
            # 输出通道是N
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)  # 在指定网络层执行完backward（）之后调用钩子函数

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):  # x: (b,c,h,w)
        offset = self.p_conv(x)  # (b,2N,h,w) 学习到的偏移量 2N表示在x轴方向的偏移和在y轴方向的偏移
        if self.modulation:  # 如果需要调制
            m = torch.sigmoid(self.m_conv(x))  # (b,N,h,w) 学习到的N个调制标量

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 如果需要调制
        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # (b,c,h,w,N)
            x_offset *= m  # 为偏移添加调制标量

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class DeformableAttention(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:  # 是否调制
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lra)  # 在指定网络层执行完backward()之后调用钩子函数

            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lra(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)
        out = torch.cat([max_out, avg_out], dim=1)
        # out = torch.cat([avg_out, max_out], dim=1)
        # out = self.conv(out)

        # 如果需要调制
        if self.distortionmode:
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))  # (b,N,h,w) 学习到的N个调制标量,试试out换成x
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)

            # out = d * out  # 为偏移添加调制标量
        out = self.conv(out)
        # mask = self.sigmoid(out)
        mask = self.sigmoid(self.upsample(out))
        att_out = x * mask
        return F.relu(att_out)


class DeformableAttention2(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableAttention2, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:  # 是否调制
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lrb)  # 在指定网络层执行完backward()之后调用钩子函数

            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrn)

    @staticmethod
    def _set_lrb(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    @staticmethod
    def _set_lrn(module, grad_input, grad_output):
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)
        out = torch.cat([max_out, avg_out], dim=1)
        # out = torch.cat([avg_out, max_out], dim=1)
        # out = self.conv(out)

        # 如果需要调制
        if self.distortionmode:
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))  # (b,N,h,w) 学习到的N个调制标量,试试out换成x
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)

            # out = d * out  # 为偏移添加调制标量
        out = self.conv(out)
        # mask = self.sigmoid(out)
        mask = self.sigmoid(self.upsample(out))
        att_out = x * mask
        return F.relu(att_out)


class DeformableCh_Attention(nn.Module):
    def __init__(self, stride, channels, distortionmode=False):
        super(DeformableCh_Attention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // 16, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        if distortionmode:  # 是否调制
            self.d_conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, stride=stride, bias=False)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lrs)  # 在指定网络层执行完backward()之后调用钩子函数

            # self.d_conv1 = nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride)
            # nn.init.constant_(self.d_conv1.weight, 0)
            # self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lrs(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
        # print(module)
        # print('grad_input: ', grad_input)
        # print('grad_output: ', grad_output)

    # @staticmethod
    # def _set_lrm(module, grad_input, grad_output):
    #     grad_input = (grad_input[i] * 0.5 for i in range(len(grad_input)))
    #     grad_output = (grad_output[i] * 0.5 for i in range(len(grad_output)))

    def forward(self, x):

        avg_out = self.avg_pool(x)

        # 如果需要调制
        if self.distortionmode:
            d_out = torch.sigmoid(self.d_conv(avg_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
            # d_avg_out = torch.sigmoid(self.d_conv(avg_out))  # (b,N,h,w) 学习到的N个调制标量,试试out换成x
            # d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out = d_out * avg_out    # 为偏移添加调制标量

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        mask = self.sigmoid(out)
        att_out = x * mask
        return att_out
