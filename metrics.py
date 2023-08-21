import torch
import numpy as np
import math
from torch import nn
from tools.bicubic_torch import imresize as resize
from tools.cross_correlation import xcorr_torch
from coregistration import fineshift


class DowngradeProtocol(nn.Module):
    def __init__(self, mtf, ratio, device):
        super(DowngradeProtocol, self).__init__()

        # Parameters definition
        kernel = mtf
        self.pad_size = math.floor((kernel.shape[0] - 1) / 2)
        nbands = kernel.shape[-1]
        self.ratio = ratio
        self.device = device
        # Conversion of filters in Tensor
        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.pad = nn.ReplicationPad2d(self.pad_size)

    def forward(self, outputs, r, c):

        x = self.pad(outputs)
        x = self.depthconv(x)
        xx = []
        for bs in range(x.shape[0]):
            xx.append(fineshift(torch.unsqueeze(x[bs], 0), r[bs], c[bs], self.device))
        x = torch.cat(xx, 0)
        x = x[:, :, 2::self.ratio, 2::self.ratio]

        return x


def normalize_block(im):
    m = im.view(im.shape[0], im.shape[1], -1).mean(2).view(im.shape[0], im.shape[1], 1, 1)
    s = im.view(im.shape[0], im.shape[1], -1).std(2).view(im.shape[0], im.shape[1], 1, 1)

    s[s == 0] = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def cayley_dickson_property_1d(onion1, onion2):
    n = onion1.shape[1]

    if n > 1:
        middle = int(n / 2)
        a = onion1[:, :middle]
        b = onion1[:, middle:]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0] = 1
        b = b * neg
        c = onion2[:, :middle]
        d = onion2[:, middle:]
        d = d * neg

        if n == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_1d(a, c)
            ris2 = cayley_dickson_property_1d(d, b * neg)
            ris3 = cayley_dickson_property_1d(a * neg, d)
            ris4 = cayley_dickson_property_1d(c, b)

            ris = torch.cat((ris1 - ris2, ris3 + ris4), dim=1)
    else:
        ris = onion1 * onion2

    return ris


def cayley_dickson_property_2d(onion1, onion2):
    dim3 = onion1.shape[1]
    if dim3 > 1:
        middle = int(dim3 / 2)

        a = onion1[:, 0:middle, :, :]
        b = onion1[:, middle:, :, :]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0, :, :] = 1
        b = b * neg
        c = onion2[:, 0:middle, :, :]
        d = onion2[:, middle:, :, :]

        d = d * neg
        if dim3 == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d, b * neg)
            ris3 = cayley_dickson_property_2d(a * neg, d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = torch.cat((aux1, aux2), dim=1)
    else:
        ris = onion1 * onion2

    return ris


def q_index(im1, im2, size, device):
    im1 = im1.double()
    im2 = im2.double()
    neg = -torch.ones(im2.shape, dtype=im2.dtype, device=im2.device)
    neg[:, 0, :, :] = 1

    batch_size, dim3, _, _ = im1.size()

    im1, s, t = normalize_block(im1)

    condition = (s[:, 0, 0, 0] == 0)
    im2[condition] = im2[condition] - s[condition] + 1
    im2[~condition] = ((im2[~condition] - s[~condition]) / t[~condition]) + 1

    im2 = im2 * neg

    m1 = torch.mean(im1, dim=(2, 3))
    m2 = torch.mean(im2, dim=(2, 3))
    mod_q1m = torch.sqrt(torch.sum(torch.pow(m1, 2), dim=1))
    mod_q2m = torch.sqrt(torch.sum(torch.pow(m2, 2), dim=1))

    mod_q1 = torch.sqrt(torch.sum(torch.pow(im1, 2), dim=1))
    mod_q2 = torch.sqrt(torch.sum(torch.pow(im2, 2), dim=1))

    term2 = mod_q1m * mod_q2m
    term4 = torch.pow(mod_q1m, 2) + torch.pow(mod_q2m, 2)
    temp = [(size * size) / (size * size - 1)] * batch_size
    temp = torch.tensor(temp, device=device)
    int1 = torch.clone(temp)
    int2 = torch.clone(temp)
    int3 = torch.clone(temp)
    int1 = int1 * torch.mean(torch.pow(mod_q1, 2), dim=(-2, -1))
    int2 = int2 * torch.mean(torch.pow(mod_q2, 2), dim=(-2, -1))
    int3 = int3 * (torch.pow(mod_q1m, 2) + torch.pow(mod_q2m, 2))
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    condition = (term3 == 0)
    q = torch.zeros((batch_size, dim3), device=device, dtype=mean_bias.dtype, requires_grad=False)
    q[condition, dim3 - 1] = mean_bias[condition]

    cbm = 2 / term3
    qu = cayley_dickson_property_2d(im1, im2)
    qm = cayley_dickson_property_1d(m1, m2)
    qv = (size * size) / (size * size - 1) * torch.mean(qu, dim=(-2, -1))
    q[~condition] = (qv[~condition] - (temp[:, None] * qm)[~condition])[:, :]
    q[~condition] = q[~condition] * mean_bias[~condition, None] * cbm[~condition, None]

    q[q == 0] = 1e-30

    return q


class Q2n(nn.Module):
    def __init__(self, device, q_block_size=32, q_shift=32):
        super(Q2n, self).__init__()

        self.Q_block_size = q_block_size
        self.Q_shift = q_shift
        self.device = device

    def forward(self, outputs, labels):

        bs, dim3, dim1, dim2 = labels.size()

        if math.ceil(math.log2(dim1)) - math.log2(dim1) != 0:
            difference = 2 ** (math.ceil(math.log2(dim1))) - dim1
            pads_2n = nn.ReplicationPad2d((math.floor(difference / 2), math.ceil(difference / 2), 0, 0))
            labels = pads_2n(labels)
            outputs = pads_2n(outputs)

        if math.ceil(math.log2(dim2)) - math.log2(dim2) != 0:
            difference = 2 ** (math.ceil(math.log2(dim2))) - dim2
            pads_2n = nn.ReplicationPad2d((0, 0, math.floor(difference / 2), math.ceil(difference / 2)))
            labels = pads_2n(labels)
            outputs = pads_2n(outputs)

        bs, dim3, dim1, dim2 = labels.size()

        stepx = math.ceil(dim1 / self.Q_shift)
        stepy = math.ceil(dim2 / self.Q_shift)

        if stepy <= 0:
            stepy = 1
            stepx = 1

        est1 = (stepx - 1) * self.Q_shift + self.Q_block_size - dim1
        est2 = (stepy - 1) * self.Q_shift + self.Q_block_size - dim2

        if (est1 != 0) + (est2 != 0) > 0:
            padding = torch.nn.ReflectionPad2d((0, est1, 0, est2))

            reference = padding(labels)
            outputs = padding(outputs)

            outputs = torch.round(outputs)
            labels = torch.round(reference)
        bs, dim3, dim1, dim2 = labels.size()

        if math.ceil(math.log2(dim3)) - math.log2(dim3) != 0:
            exp_difference = 2 ** (math.ceil(math.log2(dim3))) - dim3
            diff = torch.zeros((bs, exp_difference, dim1, dim2), device=self.device, requires_grad=False)
            labels = torch.cat((labels, diff), dim=1)
            outputs = torch.cat((outputs, diff), dim=1)

        values = []
        for j in range(stepx):
            values_i = []
            for i in range(stepy):
                o = q_index(labels[:, :, j * self.Q_shift:j * self.Q_shift + self.Q_block_size,
                            i * self.Q_shift: i * self.Q_shift + self.Q_block_size],
                            outputs[:, :, j * self.Q_shift:j * self.Q_shift + self.Q_block_size,
                            i * self.Q_shift: i * self.Q_shift + self.Q_block_size], self.Q_block_size,
                            self.device)
                values_i.append(o[:, :, None, None])
            values_i = torch.cat(values_i, -1)
            values.append(values_i)
        values = torch.cat(values, -2)
        q2n_index_map = torch.sqrt(torch.sum(values ** 2, dim=1))
        q2n_index = torch.mean(q2n_index_map, dim=(-2, -1))

        return q2n_index


class ReproDLambdaKhan(nn.Module):
    def __init__(self, device):
        super(ReproDLambdaKhan, self).__init__()
        self.Q2n = Q2n(device)

    def forward(self, shifted_downgraded_outputs, ms):
        q2n_index = self.Q2n(shifted_downgraded_outputs, ms)
        dlambda = 1.0 - torch.mean(q2n_index)

        return dlambda


class ERGAS(nn.Module):
    def __init__(self, ratio, reduction='mean'):
        super(ERGAS, self).__init__()
        self.ratio = ratio
        self.reduction = reduction

    def forward(self, outputs, labels):
        mu = torch.mean(labels, dim=(2, 3)) ** 2
        nbands = labels.size(dim=1)
        error = torch.mean((outputs - labels) ** 2, dim=(2, 3))
        ergas_index = 100 / self.ratio * torch.sqrt(torch.sum(error / mu, dim=1) / nbands)
        if self.reduction == 'mean':
            ergas = torch.mean(ergas_index)
        else:
            ergas = torch.sum(ergas_index)

        return ergas


class SAM(nn.Module):
    def __init__(self, reduction='mean'):
        super(SAM, self).__init__()
        self.reduction = reduction

    @staticmethod
    def forward(outputs, labels):
        norm_outputs = torch.sum(outputs * outputs, dim=1)
        norm_labels = torch.sum(labels * labels, dim=1)
        scalar_product = torch.sum(outputs * labels, dim=1)
        norm_product = torch.sqrt(norm_outputs * norm_labels)
        scalar_product[norm_product == 0] = float('nan')
        norm_product[norm_product == 0] = float('nan')
        scalar_product = torch.flatten(scalar_product, 1, 2)
        norm_product = torch.flatten(norm_product, 1, 2)
        angle = torch.nansum(torch.acos(scalar_product / norm_product), dim=1) / norm_product.shape[1]
        angle = angle * 180 / np.pi
        return angle


class Q(nn.Module):
    def __init__(self, nbands, block_size=32):
        super(Q, self).__init__()
        self.block_size = block_size
        self.N = block_size ** 2
        filter_shape = (nbands, 1, self.block_size, self.block_size)
        kernel = torch.ones(filter_shape, dtype=torch.float32)

        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)
        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

    def forward(self, outputs, labels):
        outputs_sq = outputs ** 2
        labels_sq = labels ** 2
        outputs_labels = outputs * labels

        outputs_sum = self.depthconv(outputs)
        labels_sum = self.depthconv(labels)

        outputs_sq_sum = self.depthconv(outputs_sq)
        labels_sq_sum = self.depthconv(labels_sq)
        outputs_labels_sum = self.depthconv(outputs_labels)

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2
        numerator = 4 * (self.N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = self.N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)
        quality_map = torch.ones(denominator.size(), device=outputs.device)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = (denominator != 0)
        quality_map[index] = numerator[index] / denominator[index]
        quality = torch.mean(quality_map, dim=(2, 3))

        return quality


class DRho(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(DRho, self).__init__()

        # Parameters definition:

        self.scale = math.ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels):
        x_corr = torch.clamp(xcorr_torch(outputs, labels, self.scale), min=-1.0, max=1.0)
        x = 1.0 - x_corr
        d_rho = torch.mean(x)

        return d_rho


class Ds(nn.Module):
    def __init__(self, nbands, ratio=4, q=1, q_block_size=32):
        super(Ds, self).__init__()
        self.Q_high = Q(nbands, q_block_size)
        self.Q_low = Q(nbands, q_block_size // ratio)
        self.nbands = nbands
        self.ratio = ratio
        self.q = q

    def forward(self, outputs, pan, ms):
        pan = pan.repeat(1, self.nbands, 1, 1)
        pan_lr = resize(pan, scale=1 / self.ratio)

        q_high = self.Q_high(outputs, pan)
        q_low = self.Q_low(ms, pan_lr)

        ds = torch.sum(abs(q_high - q_low) ** self.q, dim=1)

        ds = (ds / self.nbands) ** (1 / self.q)

        return ds


class LSR(nn.Module):
    def __init__(self):
        # Class initialization
        super(LSR, self).__init__()

    @staticmethod
    def forward(outputs, pan):
        pan = pan.double()
        outputs = outputs.double()

        pan_flatten = torch.flatten(pan, start_dim=-2).transpose(2, 1)
        fused_flatten = torch.flatten(outputs, start_dim=-2).transpose(2, 1)
        with torch.no_grad():
            alpha = (fused_flatten.pinverse() @ pan_flatten)[:, :, :, None]
        i_r = torch.sum(outputs * alpha, dim=1, keepdim=True)

        err_reg = pan - i_r

        cd = 1 - (torch.var(err_reg, dim=(1, 2, 3)) / torch.var(pan, dim=(1, 2, 3)))

        return cd


class DsR(nn.Module):
    def __init__(self):
        super(DsR, self).__init__()
        self.metric = LSR()

    def forward(self, outputs, pan):
        lsr = torch.mean(self.metric(outputs, pan))
        return 1.0 - lsr
