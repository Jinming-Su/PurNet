import torch
import torch.nn as nn
import torch.nn.functional as fnc

import numpy as np
import time

from torch.autograd import Variable
from torch import nn

class ImprovedBCELoss(nn.Module):
    def __init__(self):
        super(ImprovedBCELoss, self).__init__()

    def forward(self, pt, gt, em):

        weighted = 1 + em.abs()
        pt = pt.clamp(min=1e-4)
        pt = pt.clamp(max=1-(1e-4))
        loss_total = (gt.mul(pt.log()) + (1 - gt).mul((1 - pt).log())).mul(weighted)
        loss_mean = loss_total.sum() / (pt.shape[0] * pt.shape[1] * pt.shape[2] * pt.shape[3])

        return -loss_mean