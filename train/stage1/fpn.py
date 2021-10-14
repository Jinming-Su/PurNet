import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from resnet import ResNet50

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        resnet = ResNet50()
        self.layer0 = resnet.layer0
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for layer_cnt in range(self.layer3.__len__()):
            layer_length = list(self.layer3[layer_cnt].children()).__len__()
            #self.layer3[layer_cnt].conv2.stride = (1, 1)
            self.layer3[layer_cnt].conv2.dilation = (2, 2)
            self.layer3[layer_cnt].conv2.padding = (2, 2)
            #if layer_length == 8:
            #    self.layer3[layer_cnt].downsample[0].stride = (1, 1)

        for layer_cnt in range(self.layer4.__len__()):
            layer_length = list(self.layer4[layer_cnt].children()).__len__()
            self.layer4[layer_cnt].conv2.stride = (1, 1)
            self.layer4[layer_cnt].conv2.dilation = (4, 4)
            self.layer4[layer_cnt].conv2.padding = (4, 4)
            if layer_length == 8:
                self.layer4[layer_cnt].downsample[0].stride = (1, 1)

        self.convert5 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.convert4 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.convert3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.convert2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.convert1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.feature_upscore5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            #nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.feature_upscore4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.feature_upscore3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.feature_upscore2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.predict5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        )
        self.predict4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        )
        self.predict2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        )
        self.predict1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout) or isinstance(m, nn.PReLU):
                m.inplace = True
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):

        layer1 = self.layer0(x)
        layer2 = self.layer1(layer1)
        layer3 = self.layer2(layer2)
        layer4 = self.layer3(layer3)
        layer5 = self.layer4(layer4)

        x_size = x.size()[2:]
        l2_size = layer2.size()[2:]
        l3_size = layer3.size()[2:]
        l4_size = layer4.size()[2:]

        feature5 = self.convert5(layer5)

        feature4 = self.feature_upscore5(feature5)
        #feature4 = feature4[:, :, 1: 1 + l4_size[0], 1:1 + l4_size[1]]
        feature4 = self.convert4(layer4) + feature4

        feature3 = self.feature_upscore4(feature4)
        feature3 = feature3[:, :, 1: 1 + l3_size[0], 1:1 + l3_size[1]]
        feature3 = self.convert3(layer3) + feature3

        feature2 = self.feature_upscore3(feature3)
        feature2 = feature2[:, :, 1: 1 + l2_size[0], 1:1 + l2_size[1]]
        feature2 = self.convert2(layer2) + feature2

        feature1 = self.feature_upscore2(feature2)
        feature1 = self.convert1(layer1) + feature1


        predict5 = self.predict5(feature5)
        predict5 = predict5[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        predict4 = self.predict4(feature4)
        predict4 = predict4[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        predict3 = self.predict3(feature3)
        predict3 = predict3[:, :, 4: 4 + x_size[0], 4:4 + x_size[1]]

        predict2 = self.predict2(feature2)
        predict2 = predict2[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]

        predict1 = self.predict1(feature1)
        predict1 = predict1[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]


        #predict_coarse =
        #x_shape = x.shape
        #torch.zeros()
        #for i in range(4):



        if self.training:
            return F.sigmoid(predict1), F.sigmoid(predict2), F.sigmoid(predict3), F.sigmoid(predict4), F.sigmoid(predict5)
        return F.sigmoid(predict1)

