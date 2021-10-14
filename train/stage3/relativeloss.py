import torch
import torch.nn as nn
import torch.nn.functional as fnc

import numpy as np
import time

from torch.autograd import Variable
from torch import nn

class RelativeLoss(nn.Module):
    def __init__(self):
        super(RelativeLoss, self).__init__()

    def forward(self, pt, gt, sp):
        sp_flatten = torch.reshape(sp, (sp.shape[0], -1))
        sp_max = torch.max(sp_flatten, dim=1)[0]
        pt = fnc.sigmoid(pt)
        loss_fn = nn.KLDivLoss(reduction='sum')
        loss = 0

        batchsize = sp.shape[0]
        for i in range(0, batchsize):

            pt_t = pt[i, 0].reshape((1, -1))
            gt_t = gt[i, 0].reshape((1, -1))
            sp_max_t = sp_max[i]
            sp_t = sp[i, 0].reshape((1, -1))[0]
            sp_num = sp_max_t + 1

            sp_t_1 = torch.ones(sp_num, pt_t[0].__len__()).cuda()

            sp_t_1_mul = torch.range(0, sp_num-1).reshape((-1, 1)).cuda()

            sp_t_index = sp_t_1 * sp_t_1_mul

            sp_index = (sp_t.float() == sp_t_index).float()

            sp_len = torch.sum(sp_index, dim=1)

            sp_len[sp_len == 0] = 1

            gt_sp_saliency = sp_index.mm(gt_t.t()).t().div(sp_len)
            pt_sp_saliency = sp_index.mm(pt_t.t()).t().div(sp_len)

            gt_rel_mat = gt_sp_saliency.t() - gt_sp_saliency
            pt_rel_mat = pt_sp_saliency.t() - pt_sp_saliency


            #gt_rel_mat = Variable(gt_rel_mat).cuda()
            #pt_rel_mat = Variable(pt_rel_mat).cuda()

            gt_rel_mat = (gt_rel_mat + 1.) / 2. 
            gt_rel_mat = gt_rel_mat / torch.sum(gt_rel_mat)
            pt_rel_mat = (pt_rel_mat + 1.) / 2.
            pt_rel_mat = pt_rel_mat / torch.sum(pt_rel_mat)

            loss += loss_fn(pt_rel_mat.log(), gt_rel_mat)

        return loss / batchsize
