import torch.nn as nn
import torch.nn.functional as func
from tools.attention import CBAM


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, pad='same', pad_mode='reflect', bn=False, act=nn.GELU()):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=pad, padding_mode=pad_mode))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class LPNN(nn.Module):
    def __init__(self, n_channels, n_features=64, kernel_size=3, pad='same', pad_mode='reflect', bias_flag=True):
        super(LPNN, self).__init__()

        self.conv_1 = nn.Conv2d(n_channels, n_features, kernel_size, bias=bias_flag, padding=pad, padding_mode=pad_mode)

        self.conv_2 = nn.Conv2d(n_features, n_features, kernel_size, bias=bias_flag, padding=pad,
                                padding_mode=pad_mode)
        self.CBAM_1 = CBAM(n_features, reduction_ratio=4, spatial=True)
        self.res_block_1 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.res_block_2 = ResBlock(n_features, kernel_size, bias=bias_flag)
        self.CBAM_2 = CBAM(n_features, reduction_ratio=4, spatial=True)
        self.conv_3 = nn.Conv2d(n_features, n_channels-1, 5, bias=bias_flag, padding=pad, padding_mode=pad_mode)

    def forward(self, inp):

        x = func.relu(self.conv_1(inp))
        x = func.relu(self.conv_2(x))
        x = self.CBAM_1(x) + x
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.CBAM_2(x) + x
        x = self.conv_3(x)
        x = x + inp[:, :-1, :, :]

        return x
