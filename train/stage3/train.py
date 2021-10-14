import datetime
import os

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

import sys
sys.path.append('../')

import joint_transforms
from config import dutstr_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir

from ImprovedBCELoss import ImprovedBCELoss


from fpn_ea_oa import FPN
from torch.backends import cudnn
import torch.nn.functional as F

from relativeloss import RelativeLoss

#from tensorboardX import SummaryWriter
#writer = SummaryWriter()

import time

cudnn.benchmark = True

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'

args = {
    'net': 'restnet50_dilation',
    'resize': [320, 320],  # h, w
    'max_iter': 200000,
    'iteration_of_epoch': 10553,
    'train_batch_size': 2,
    'save_interval': 5000,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'display': 10,
}

joint_transform = joint_transforms.Compose([
    joint_transforms.Resize(args['resize']),
    #joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4923, 0.4634, 0.3974], [0.2737, 0.2659, 0.2836])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(dutstr_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True, drop_last=True)

criterion = nn.BCEWithLogitsLoss().cuda()
criterion_improved = ImprovedBCELoss().cuda()
criterion_bce = nn.BCELoss(reduction='mean').cuda()
log_path = os.path.join(str(datetime.datetime.now()) + '.txt')
criterion_rl = RelativeLoss().cuda()


def main():
    net = FPN().cuda().train()

    input_data = torch.rand(args['train_batch_size'], 3, args['resize'][0], args['resize'][1])
    #writer.add_graph(FCN().train(), (input_data,))

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[:5] == 'layer' and name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:5] == 'layer' and name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']},
	    {'params': [param for name, param in net.named_parameters() if name[:5] != 'layer' and name[:3] != 'ea_' and name[:3] != 'oa_' and name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:5] != 'layer' and name[:3] != 'ea_' and name[:3] != 'oa_' and name[-4:] != 'bias'],
         'lr': 1 *args['lr'], 'weight_decay': args['weight_decay']},
    ], momentum=args['momentum'])

    optimizer_e = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[:3] == 'ea_' and name[-4:] == 'bias'],
         'lr': 20 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:3] == 'ea_' and name[-4:] != 'bias'],
         'lr': 10 * args['lr'], 'weight_decay': args['weight_decay']},
    ], momentum=args['momentum'])

    optimizer_o = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[:3] == 'oa_' and name[-4:] == 'bias'],
         'lr': 20 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:3] == 'oa_' and name[-4:] != 'bias'],
         'lr': 10 * args['lr'], 'weight_decay': args['weight_decay']},
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer, optimizer_o, optimizer_e)


def train(net, optimizer, optimizer_o, optimizer_e):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, total_loss_o_record = AvgMeter(), AvgMeter()

        start_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                            ) ** args['lr_decay']

            inputs, labels, sps = data

            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            sps = Variable(sps).cuda()


            outputs1, outputs2, outputs3, outputs4, outputs5, \
                oa_outputs1, oa_outputs2, oa_outputs3, oa_outputs4, oa_outputs5, \
                ea_outputs1_g, ea_outputs2_g, ea_outputs3_g, ea_outputs4_g, ea_outputs5_g, \
                ea_outputs1_e, ea_outputs2_e, ea_outputs3_e, ea_outputs4_e, ea_outputs5_e \
                = net(inputs)

            loss1 = criterion_improved(outputs1, labels, ea_outputs1_e)
            loss2 = criterion_improved(outputs2, labels, ea_outputs2_e)
            loss3 = criterion_improved(outputs3, labels, ea_outputs3_e)
            loss4 = criterion_improved(outputs4, labels, ea_outputs4_e)
            loss5 = criterion_improved(outputs5, labels, ea_outputs5_e)

            oa_loss1 = criterion_bce(oa_outputs1, labels)
            oa_loss2 = criterion_bce(oa_outputs2, labels)
            oa_loss3 = criterion_bce(oa_outputs3, labels)
            oa_loss4 = criterion_bce(oa_outputs4, labels)
            oa_loss5 = criterion_bce(oa_outputs5, labels)

            ea_loss1_g = criterion_bce(ea_outputs1_g, labels)
            ea_loss2_g = criterion_bce(ea_outputs2_g, labels)
            ea_loss3_g = criterion_bce(ea_outputs3_g, labels)
            ea_loss4_g = criterion_bce(ea_outputs4_g, labels)
            ea_loss5_g = criterion_bce(ea_outputs5_g, labels)

            ea_outputs1_g = Variable(ea_outputs1_g).cuda()
            ea_outputs2_g = Variable(ea_outputs2_g).cuda()
            ea_outputs3_g = Variable(ea_outputs3_g).cuda()
            ea_outputs4_g = Variable(ea_outputs4_g).cuda()
            ea_outputs5_g = Variable(ea_outputs5_g).cuda()

            ea_loss1_e = criterion_bce((ea_outputs1_e + 1) / 2, (labels - ea_outputs1_g + 1) / 2)
            ea_loss2_e = criterion_bce((ea_outputs2_e + 1) / 2, (labels - ea_outputs2_g + 1) / 2)
            ea_loss3_e = criterion_bce((ea_outputs3_e + 1) / 2, (labels - ea_outputs3_g + 1) / 2)
            ea_loss4_e = criterion_bce((ea_outputs4_e + 1) / 2, (labels - ea_outputs4_g + 1) / 2)
            ea_loss5_e = criterion_bce((ea_outputs5_e + 1) / 2, (labels - ea_outputs5_g + 1) / 2)

            loss1_rl = criterion_rl(outputs1, labels, sps)
            loss2_rl = criterion_rl(outputs2, labels, sps)
            loss3_rl = criterion_rl(outputs3, labels, sps)
            loss4_rl = criterion_rl(outputs4, labels, sps)
            loss5_rl = criterion_rl(outputs5, labels, sps)

            optimizer.zero_grad()
            total_loss_rl = loss1_rl + loss2_rl + loss3_rl + loss4_rl + loss5_rl
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + total_loss_rl
            #writer.add_image('image', inputs[0].squeeze())
            #writer.add_image('output', nn.functional.sigmoid(outputs1[0]))
            #writer.add_image('label', labels[0])
            total_loss.backward(retain_graph=True)
            optimizer.step()

            optimizer_o.zero_grad()
            total_loss_o = oa_loss1 + oa_loss2 + oa_loss3 + oa_loss4 + oa_loss5
            total_loss_o.backward(retain_graph=True)
            optimizer_o.step()

            optimizer_e.zero_grad()
            total_loss_e = ea_loss1_e + ea_loss2_e + ea_loss3_e + ea_loss4_e + ea_loss5_e + \
				ea_loss1_g + ea_loss2_g + ea_loss3_g + ea_loss4_g + ea_loss5_g
            total_loss_e.backward()
            optimizer_e.step()

            total_loss_record.update(total_loss.item(), batch_size)
            total_loss_o_record.update(total_loss_o.item(), batch_size)

            curr_iter += 1

            if curr_iter % args['display'] == 0:
                #writer.add_scalar('loss', total_loss, global_step=i)
                total_time = time.time()-start_time
                rest_time = (total_time * 1.0 / args['display'])*(args['max_iter'] - curr_iter)
                start_time = time.time()

                curr_epoch = curr_iter * args['train_batch_size'] / args['iteration_of_epoch'] + 1

                log = '%s : e %d | iter %d | total l %.3f(%.3f) |tl %.3f(%.3f) | l0 %.3f(%.3f) | l1 %.3f(%.3f) | l2 %.3f(%.3f) | l3 %.3f(%.3f) | l4 %.3f(%.3f) | g0 %.3f | g1 %.3f | g2 %.3f | g3 %.3f | g4 %.3f | e0 %.3f | e1 %.3f | e2 %.3f | e3 %.3f | e4 %.3f | lr %.5f | t/s %.2f  rest/s %d:%d:%d' % \
                      (str(datetime.datetime.now()), curr_epoch, curr_iter, total_loss_record.avg, total_loss_o_record.avg, total_loss, total_loss_o, \
                       loss1, oa_loss1, loss2, oa_loss2, loss3, oa_loss3, loss4, oa_loss4, loss5, oa_loss5,\
                       ea_loss1_g, ea_loss2_g, ea_loss3_g, ea_loss4_g, ea_loss5_g, \
                       ea_loss1_e, ea_loss2_e, ea_loss3_e, ea_loss4_e, ea_loss5_e, \
                       optimizer.param_groups[1]['lr'], total_time, \
                       int(rest_time/3600), int(rest_time%3600/60), int(rest_time%60))

                print(log)
                open(log_path, 'a').write(log + '\n')


            if curr_iter % args['save_interval'] == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, '%d_optim.pth' % curr_iter))
            if curr_iter >=  args['max_iter']:
                return 

if __name__ == '__main__':
    main()
