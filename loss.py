import torch.nn as nn
import torch
from math import ceil
from tools.cross_correlation import xcorr_torch as ccorr


class StructuralLoss(nn.Module):

    def __init__(self, sigma):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:
        self.scale = ceil(sigma / 2)

    def forward(self, outputs, labels, xcorr_thr):
        x_corr = torch.clamp(ccorr(outputs, labels, self.scale), min=-1)
        x = 1.0 - x_corr

        with torch.no_grad():
            loss_cross_corr_wo_thr = torch.mean(x)

        worst = x.gt(xcorr_thr)
        y = x * worst
        loss_cross_corr = torch.mean(y)

        return loss_cross_corr, loss_cross_corr_wo_thr.item()
