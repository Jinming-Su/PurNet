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

from fpn import FPN
from torch.backends import cudnn
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
    'max_iter': 50000,
    'iteration_of_epoch': 10553,
    'train_batch_size': 4,
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

criterion_bce = nn.BCELoss(reduction='mean').cuda()
criterion_rl = RelativeLoss().cuda()
log_path = os.path.join(str(datetime.datetime.now()) + '.txt')


def main():
    net = FPN().cuda().train()

    input_data = torch.rand(args['train_batch_size'], 3, args['resize'][0], args['resize'][1])
    #writer.add_graph(FCN().train(), (input_data,))

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[:5] == 'layer' and name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:5] == 'layer' and name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']},
	    {'params': [param for name, param in net.named_parameters() if name[:5] != 'layer' and name[-4:] == 'bias'],
         'lr': 20 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[:5] != 'layer' and name[-4:] != 'bias'],
         'lr': 10 *args['lr'], 'weight_decay': args['weight_decay']},
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
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, total_loss_rl_record = AvgMeter(), AvgMeter()

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

            optimizer.zero_grad()
            outputs1, outputs2, outputs3, outputs4, outputs5 = net(inputs)
            loss1 = criterion_bce(outputs1, labels)
            loss2 = criterion_bce(outputs2, labels)
            loss3 = criterion_bce(outputs3, labels)
            loss4 = criterion_bce(outputs4, labels)
            loss5 = criterion_bce(outputs5, labels)

            loss1_rl = criterion_rl(outputs1, labels, sps)
            loss2_rl = criterion_rl(outputs2, labels, sps)
            loss3_rl = criterion_rl(outputs3, labels, sps)
            loss4_rl = criterion_rl(outputs4, labels, sps)
            loss5_rl = criterion_rl(outputs5, labels, sps)

            total_loss_rl = loss1_rl + loss2_rl + loss3_rl + loss4_rl + loss5_rl
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + total_loss_rl

            #writer.add_image('image', inputs[0].squeeze())
            #writer.add_image('output', nn.functional.sigmoid(outputs1[0]))
            #writer.add_image('label', labels[0])

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            total_loss_rl_record.update(total_loss_rl.item(), batch_size)


            curr_iter += 1

            if curr_iter % args['display'] == 0:
                #writer.add_scalar('loss', total_loss, global_step=i)
                total_time = time.time()-start_time
                rest_time = (total_time * 1.0 / args['display'])*(args['max_iter'] - curr_iter)
                start_time = time.time()

                curr_epoch = curr_iter * args['train_batch_size'] / args['iteration_of_epoch'] + 1

                log = '%s : e %d | iter %d | total l %.3f(%.3f) |tl %.3f(%.3f) | l0 %.3f(%.3f) | l1 %.3f(%.3f) | l2 %.3f(%.3f) | l3 %.3f(%.3f) | l4 %.3f(%.3f) |  lr %.5f | t/s %.2f  rest/s %d:%d:%d' % \
                      (str(datetime.datetime.now()), curr_epoch, curr_iter, total_loss_record.avg, total_loss_rl_record.avg, total_loss, total_loss_rl, \
                       loss1, loss1_rl, loss2, loss2_rl, loss3, loss3_rl, loss4, loss4_rl, loss5, loss5_rl, optimizer.param_groups[1]['lr'], total_time, \
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
