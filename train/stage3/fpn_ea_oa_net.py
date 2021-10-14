import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from resnet import ResNet50
import fpn_ea_oa

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
        #resnet = ResNet50()
        fpn_resnet = fpn_ea_oa.FPN()
        fpn_resnet.load_state_dict(torch.load('ckpt/stage2.pth'))
        self.layer0 = fpn_resnet.layer0
        self.layer1 = fpn_resnet.layer1
        self.layer2 = fpn_resnet.layer2
        self.layer3 = fpn_resnet.layer3
        self.layer4 = fpn_resnet.layer4

        #for layer_cnt in range(self.layer3.__len__()):
        #    layer_length = list(self.layer3[layer_cnt].children()).__len__()
        #    #self.layer3[layer_cnt].conv2.stride = (1, 1)
        #    self.layer3[layer_cnt].conv2.dilation = (2, 2)
        #    self.layer3[layer_cnt].conv2.padding = (2, 2)
        #    #if layer_length == 8:
        #    #    self.layer3[layer_cnt].downsample[0].stride = (1, 1)

        #for layer_cnt in range(self.layer4.__len__()):
        #    layer_length = list(self.layer4[layer_cnt].children()).__len__()
        #    self.layer4[layer_cnt].conv2.stride = (1, 1)
        #    self.layer4[layer_cnt].conv2.dilation = (4, 4)
        #    self.layer4[layer_cnt].conv2.padding = (4, 4)
        #    if layer_length == 8:
        #        self.layer4[layer_cnt].downsample[0].stride = (1, 1)

        ######## mid
        #self.convert5 = nn.Sequential(
        #    nn.Conv2d(2048, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #)
        #self.convert4 = nn.Sequential(
        #    nn.Conv2d(1024, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #)
        #self.convert3 = nn.Sequential(
        #    nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #)
        #self.convert2 = nn.Sequential(
        #    nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #)
        #self.convert1 = nn.Sequential(
        #    nn.Conv2d(64, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #)
        self.convert5 = fpn_resnet.convert5
        self.convert4 = fpn_resnet.convert4
        self.convert3 = fpn_resnet.convert3
        self.convert2 = fpn_resnet.convert2
        self.convert1 = fpn_resnet.convert1

        #self.feature_upscore5 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    #nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        #)
        #self.feature_upscore4 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        #)
        #self.feature_upscore3 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        #)
        #self.feature_upscore2 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #)
        self.feature_upscore5 = fpn_resnet.feature_upscore5
        self.feature_upscore4 = fpn_resnet.feature_upscore4
        self.feature_upscore3 = fpn_resnet.feature_upscore3
        self.feature_upscore2 = fpn_resnet.feature_upscore2

        #self.predict5 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 1, kernel_size=1),
        #    nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        #)
        #self.predict4 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 1, kernel_size=1),
        #    nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        #)
        #self.predict3 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 1, kernel_size=1),
        #    nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        #)
        #self.predict2 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 1, kernel_size=1),
        #    nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        #)
        #self.predict1 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 1, kernel_size=1),
        #    nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        #)
        self.predict5 = fpn_resnet.predict5
        self.predict4 = fpn_resnet.predict4
        self.predict3 = fpn_resnet.predict3
        self.predict2 = fpn_resnet.predict2
        self.predict1 = fpn_resnet.predict1

        self.feature_upscore5_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.feature_upscore4_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.feature_upscore3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.feature_upscore2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.feature_upscore1_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )

        ######## error attention
        self.ea_convert5 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_convert4 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_convert3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_convert2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_convert1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )


        self.ea_feature_upscore5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            # nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.ea_feature_upscore4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.ea_feature_upscore3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.ea_feature_upscore2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.ea_pre_predict5_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_e_pre_predict5_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.ea_pre_predict4_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_e_pre_predict4_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.ea_pre_predict3_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_e_pre_predict3_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.ea_pre_predict2_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_e_pre_predict2_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.ea_pre_predict1_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.ea_e_pre_predict1_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )


        self.ea_predict5_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.ea_e_predict5_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.ea_predict4_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.ea_e_predict4_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.ea_predict3_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.ea_e_predict3_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.ea_predict2_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.ea_e_predict2_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.ea_predict1_g = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.ea_e_predict1_e = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.ea_attention5_e_tanh = nn.Tanh()
        self.ea_attention4_e_tanh = nn.Tanh()
        self.ea_attention3_e_tanh = nn.Tanh()
        self.ea_attention2_e_tanh = nn.Tanh()
        self.ea_attention1_e_tanh = nn.Tanh()

        self.ea_predict5_g_upscore = nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        self.ea_e_predict5_e_upscore = nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        self.ea_predict4_g_upscore = nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        self.ea_e_predict4_e_upscore = nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        self.ea_predict3_g_upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.ea_e_predict3_e_upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.ea_predict2_g_upscore = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.ea_e_predict2_e_upscore = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.ea_predict1_g_upscore = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.ea_e_predict1_e_upscore = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)

        ######## object attention
        self.oa_convert5 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.oa_convert4 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.oa_convert3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.oa_convert2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.oa_convert1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.oa_feature_upscore5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            # nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.oa_feature_upscore4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.oa_feature_upscore3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        )
        self.oa_feature_upscore2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        )

        self.oa_predict5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        )
        self.oa_predict4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 32, stride=16, bias=False)
        )
        self.oa_predict3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        )
        self.oa_predict2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        )
        self.oa_predict1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        )

        self.oa_attention1 = Attention(128, 80, 80)
        self.oa_attention2 = Attention(128, 80, 80)
        self.oa_attention3 = Attention(128, 40, 40)
        self.oa_attention4 = Attention(128, 20, 20)
        self.oa_attention5 = Attention(128, 20, 20)

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout) or isinstance(m, nn.PReLU):
                m.inplace = True
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        self.load_state_dict(torch.load('ckpt/stage2.pth'))

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

        ##### object attention
        oa_feature5 = self.oa_convert5(layer5)

        oa_feature4 = self.oa_feature_upscore5(oa_feature5)
        oa_feature4 = self.oa_convert4(layer4) + oa_feature4

        oa_feature3 = self.oa_feature_upscore4(oa_feature4)
        oa_feature3 = oa_feature3[:, :, 1: 1 + l3_size[0], 1:1 + l3_size[1]]
        oa_feature3 = self.oa_convert3(layer3) + oa_feature3

        oa_feature2 = self.oa_feature_upscore3(oa_feature3)
        oa_feature2 = oa_feature2[:, :, 1: 1 + l2_size[0], 1:1 + l2_size[1]]
        oa_feature2 = self.oa_convert2(layer2) + oa_feature2

        oa_feature1 = self.oa_feature_upscore2(oa_feature2)
        oa_feature1 = self.oa_convert1(layer1) + oa_feature1

        oa_attention5 = self.oa_attention5(oa_feature5)
        oa_predict5 = oa_attention5 * oa_feature5 + oa_feature5
        oa_predict5 = self.oa_predict5(oa_predict5)
        oa_predict5 = oa_predict5[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        oa_attention4 = self.oa_attention4(oa_feature4)
        oa_predict4 = oa_attention4 * oa_feature4 + oa_feature4
        oa_predict4 = self.oa_predict4(oa_predict4)
        oa_predict4 = oa_predict4[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        oa_attention3 = self.oa_attention3(oa_feature3)
        oa_predict3 = oa_attention3 * oa_feature3 + oa_feature3
        oa_predict3 = self.oa_predict3(oa_predict3)
        oa_predict3 = oa_predict3[:, :, 4: 4 + x_size[0], 4:4 + x_size[1]]

        oa_attention2 = self.oa_attention2(oa_feature2)
        oa_predict2 = oa_attention2 * oa_feature2 + oa_feature2
        oa_predict2 = self.oa_predict2(oa_predict2)
        oa_predict2 = oa_predict2[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]

        oa_attention1 = self.oa_attention1(oa_feature1)
        oa_predict1 = oa_attention1 * oa_feature1 + oa_feature1
        oa_predict1 = self.oa_predict1(oa_predict1)
        oa_predict1 = oa_predict1[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]

        ##### error attention
        ea_feature5 = self.ea_convert5(layer5)

        ea_feature4 = self.ea_feature_upscore5(ea_feature5)
        ea_feature4 = self.ea_convert4(layer4) + ea_feature4

        ea_feature3 = self.ea_feature_upscore4(ea_feature4)
        ea_feature3 = ea_feature3[:, :, 1: 1 + l3_size[0], 1:1 + l3_size[1]]
        ea_feature3 = self.ea_convert3(layer3) + ea_feature3

        ea_feature2 = self.ea_feature_upscore3(ea_feature3)
        ea_feature2 = ea_feature2[:, :, 1: 1 + l2_size[0], 1:1 + l2_size[1]]
        ea_feature2 = self.ea_convert2(layer2) + ea_feature2

        ea_feature1 = self.ea_feature_upscore2(ea_feature2)
        ea_feature1 = self.ea_convert1(layer1) + ea_feature1

        ea_attention5_g = self.ea_pre_predict5_g(ea_feature5)
        ea_attention5_e = self.ea_e_pre_predict5_e(ea_feature5) - ea_attention5_g
        ea_attention5_e_tanh = self.ea_attention5_e_tanh(ea_attention5_e)

        ea_predict5_g = ea_attention5_g * ea_attention5_e_tanh + ea_attention5_g
        ea_predict5_g = self.ea_predict5_g(ea_predict5_g)
        ea_predict5_g_upscore = self.ea_predict5_g_upscore(ea_predict5_g)
        ea_predict5_g_upscore = ea_predict5_g_upscore[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]
        ea_predict5_e = self.ea_e_predict5_e(ea_attention5_e)
        ea_predict5_e_upscore = self.ea_e_predict5_e_upscore(ea_predict5_e)
        ea_predict5_e_upscore = ea_predict5_e_upscore[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        ea_attention4_g = self.ea_pre_predict4_g(ea_feature4)
        ea_attention4_e = self.ea_e_pre_predict4_e(ea_feature4) - ea_attention4_g
        ea_attention4_e_tanh = self.ea_attention4_e_tanh(ea_attention4_e)

        ea_predict4_g = ea_attention4_g * ea_attention4_e_tanh + ea_attention4_g
        ea_predict4_g = self.ea_predict4_g(ea_predict4_g)
        ea_predict4_g_upscore = self.ea_predict4_g_upscore(ea_predict4_g)
        ea_predict4_g_upscore = ea_predict4_g_upscore[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]
        ea_predict4_e = self.ea_e_predict4_e(ea_attention4_e)
        ea_predict4_e_upscore = self.ea_e_predict4_e_upscore(ea_predict4_e)
        ea_predict4_e_upscore = ea_predict4_e_upscore[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        ea_attention3_g = self.ea_pre_predict3_g(ea_feature3)
        ea_attention3_e = self.ea_e_pre_predict3_e(ea_feature3) - ea_attention3_g
        ea_attention3_e_tanh = self.ea_attention3_e_tanh(ea_attention3_e)

        ea_predict3_g = ea_attention3_g * ea_attention3_e_tanh + ea_attention3_g
        ea_predict3_g = self.ea_predict3_g(ea_predict3_g)
        ea_predict3_g_upscore = self.ea_predict3_g_upscore(ea_predict3_g)
        ea_predict3_g_upscore = ea_predict3_g_upscore[:, :, 4: 4 + x_size[0], 4:4 + x_size[1]]
        ea_predict3_e = self.ea_e_predict3_e(ea_attention3_e)
        ea_predict3_e_upscore = self.ea_e_predict3_e_upscore(ea_predict3_e)
        ea_predict3_e_upscore = ea_predict3_e_upscore[:, :, 4: 4 + x_size[0], 4:4 + x_size[1]]

        ea_attention2_g = self.ea_pre_predict2_g(ea_feature2)
        ea_attention2_e = self.ea_e_pre_predict2_e(ea_feature2) - ea_attention2_g
        ea_attention2_e_tanh = self.ea_attention2_e_tanh(ea_attention2_e)

        ea_predict2_g = ea_attention2_g * ea_attention2_e_tanh + ea_attention2_g
        ea_predict2_g = self.ea_predict2_g(ea_predict2_g)
        ea_predict2_g_upscore = self.ea_predict2_g_upscore(ea_predict2_g)
        ea_predict2_g_upscore = ea_predict2_g_upscore[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]
        ea_predict2_e = self.ea_e_predict2_e(ea_attention2_e)
        ea_predict2_e_upscore = self.ea_e_predict2_e_upscore(ea_predict2_e)
        ea_predict2_e_upscore = ea_predict2_e_upscore[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]

        ea_attention1_g = self.ea_pre_predict1_g(ea_feature1)
        ea_attention1_e = self.ea_e_pre_predict1_e(ea_feature1) - ea_attention1_g
        ea_attention1_e_tanh = self.ea_attention1_e_tanh(ea_attention1_e)

        ea_predict1_g = ea_attention1_g * ea_attention1_e_tanh + ea_attention1_g
        ea_predict1_g = self.ea_predict1_g(ea_predict1_g)
        ea_predict1_g_upscore = self.ea_predict1_g_upscore(ea_predict1_g)
        ea_predict1_g_upscore = ea_predict1_g_upscore[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]
        ea_predict1_e = self.ea_e_predict1_e(ea_attention1_e)
        ea_predict1_e_upscore = self.ea_e_predict1_e_upscore(ea_predict1_e)
        ea_predict1_e_upscore = ea_predict1_e_upscore[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]

        ##### mid
        feature5 = self.convert5(layer5)

        feature4 = self.feature_upscore5(feature5)
        feature4 = self.convert4(layer4) + feature4

        feature3 = self.feature_upscore4(feature4)
        feature3 = feature3[:, :, 1: 1 + l3_size[0], 1:1 + l3_size[1]]
        feature3 = self.convert3(layer3) + feature3

        feature2 = self.feature_upscore3(feature3)
        feature2 = feature2[:, :, 1: 1 + l2_size[0], 1:1 + l2_size[1]]
        feature2 = self.convert2(layer2) + feature2

        feature1 = self.feature_upscore2(feature2)
        feature1 = self.convert1(layer1) + feature1

        predict5 = feature5 * oa_attention5 + feature5
        predict5 = self.feature_upscore5_2(predict5)

        predict5 = predict5 * ea_attention5_e_tanh + predict5
        predict5 = self.predict5(predict5)
        predict5 = predict5[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        predict4 = feature4 * oa_attention4 + feature4
        predict4 = self.feature_upscore4_2(predict4)

        predict4 = predict4 * ea_attention4_e_tanh + predict4
        predict4 = self.predict4(predict4)
        predict4 = predict4[:, :, 8: 8 + x_size[0], 8:8 + x_size[1]]

        predict3 = feature3 * oa_attention3 + feature3
        predict3 = self.feature_upscore3_2(predict3)

        predict3 = predict3 * ea_attention3_e_tanh + predict3
        predict3 = self.predict3(predict3)
        predict3 = predict3[:, :, 4: 4 + x_size[0], 4:4 + x_size[1]]

        predict2 = feature2 * oa_attention2 + feature2
        predict2 = self.feature_upscore2_2(predict2)

        predict2 = predict2 * ea_attention2_e_tanh + predict2
        predict2 = self.predict2(predict2)
        predict2 = predict2[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]

        predict1 = feature1 * oa_attention1 + feature1
        predict1 = self.feature_upscore1_2(predict1)

        predict1 = predict1 * ea_attention1_e_tanh + predict1
        predict1 = self.predict1(predict1)
        predict1 = predict1[:, :, 2: 2 + x_size[0], 2:2 + x_size[1]]


        #predict_coarse =
        #x_shape = x.shape
        #torch.zeros()
        #for i in range(4):



        if self.training:
            return F.sigmoid(predict1), F.sigmoid(predict2), F.sigmoid(predict3), F.sigmoid(predict4), F.sigmoid(predict5), \
                   F.sigmoid(oa_predict1), F.sigmoid(oa_predict2), F.sigmoid(oa_predict3), F.sigmoid(oa_predict4), F.sigmoid(oa_predict5), \
                   F.sigmoid(ea_predict1_g_upscore), F.sigmoid(ea_predict2_g_upscore), F.sigmoid(ea_predict3_g_upscore), F.sigmoid(ea_predict4_g_upscore), F.sigmoid(ea_predict5_g_upscore), \
                   F.tanh(ea_predict1_e_upscore), F.tanh(ea_predict2_e_upscore), F.tanh(ea_predict3_e_upscore), F.tanh(ea_predict4_e_upscore), F.tanh(ea_predict5_e_upscore)
        #return F.sigmoid(predict1), oa_attention1, ea_attention1_e_tanh, F.sigmoid(oa_predict1), F.sigmoid(ea_predict1_g_upscore), F.tanh(ea_predict1_e_upscore)
        return F.sigmoid(predict1), F.sigmoid(predict2), F.sigmoid(predict3), F.sigmoid(predict4), F.sigmoid(predict5)

class Attention(nn.Module):
    def __init__(self, channel, height, width):
        super(Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AvgPool2d((height, width)),
            nn.Softmax(dim=1)
        )

        self.spatial_attention = nn.Softmax(dim=2)

        #self.conv = nn.Sequential(
        #    nn.Conv2d(channel, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128), nn.BatchNorm2d(128), nn.PReLU(),
        #    nn.Conv2d(128, channel, kernel_size=1, padding=0), nn.BatchNorm2d(channel), nn.PReLU(),
        #)

    def forward(self, x):
        ca = self.channel_attention(x)

        temp = x.view([x.size(0), x.size(1), -1])
        sa = self.spatial_attention(temp)
        sa = sa.view([x.size(0), x.size(1), x.size(2), x.size(3)])

        x_attention = sa * ca

        #return self.conv(x_attention) + x_attention
        return x_attention
