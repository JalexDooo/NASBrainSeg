import os
import torch
import torch.nn as nn
from models.base_model import BaseModel, init_net, get_norm_layer, GeneralizedDiceLoss, expand_target
from monai.losses import DiceCELoss, DiceLoss


class li2019(nn.Module):
    '''2D Method

    Paper: https://www.sciencedirect.com/science/article/pii/S0010482519300873
    
    Note: The paper was't provided the parameter of `depth` and `channels`.

    `Channel`: 4, 32, 64, ...
    `Depth`: channel

    # flops & params: 5.505G, 4.090M, 0.08040809631347656 s
    
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(li2019, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

        self.layer1 = InceptionBlock(4, 32, 32)
        self.down1 = nn.Conv2d(4+int(32*7/8), 32, kernel_size=3, stride=2, padding=1)
        self.layer2 = InceptionBlock(32, 64, 64)
        self.down2 = nn.Conv2d(32+int(64*7/8), 64, kernel_size=3, stride=2, padding=1)
        self.layer3 = InceptionBlock(64, 128, 128)
        self.down3 = nn.Conv2d(64+int(128*7/8), 128, kernel_size=3, stride=2, padding=1)
        self.layer4 = InceptionBlock(128, 256, 256)
        self.down4 = nn.Conv2d(128+int(256*7/8), 256, kernel_size=3, stride=2, padding=1)
        self.layer5 = InceptionBlock(256, 512, 512) # out: 256+int(512*7/8)

        self.up1 = nn.ConvTranspose2d(256+int(512*7/8), 128+int(256*7/8), kernel_size=2, stride=2, output_padding=0)
        self.layer11 = InceptionBlock(128+int(256*7/8), None, 256)
        self.channelfix1 = nn.Conv2d(576, 352, kernel_size=1, stride=1, padding=0)

        self.up2 = nn.ConvTranspose2d(352, 64+int(128*7/8), kernel_size=2, stride=2, output_padding=0)
        self.layer22 = InceptionBlock(64+int(128*7/8), None, 128)
        self.upskip2 = nn.ConvTranspose2d(352, 64+int(128*7/8), kernel_size=2, stride=2, output_padding=0)
        self.channelfix2 = nn.Conv2d(288, 176, kernel_size=1, stride=1, padding=0)

        self.up3 = nn.ConvTranspose2d(176, 32+int(64*7/8), kernel_size=2, stride=2, output_padding=0)
        self.layer33 = InceptionBlock(32+int(64*7/8), None, 64)
        self.upskip3 = nn.ConvTranspose2d(176, 32+int(64*7/8), kernel_size=2, stride=2, output_padding=0)
        self.channelfix3 = nn.Conv2d(144, 88, kernel_size=1, stride=1, padding=0)

        self.up4 = nn.ConvTranspose2d(88, 32, kernel_size=2, stride=2, output_padding=0)
        self.layer44 = InceptionBlock(32, None, 32)
        self.upskip4 = nn.ConvTranspose2d(88, 32, kernel_size=2, stride=2, output_padding=0)
        self.channelfix4 = nn.Conv2d(60, 32, kernel_size=1, stride=1, padding=0)

        self.output = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # down
        x1 = self.layer1(x)
        tmp = self.down1(x1) # tmp.shape:  torch.Size([1, 32, 96, 96])
        # print('tmp.shape: ', tmp.shape)
        x2 = self.layer2(tmp) # torch.Size([1, 88, 96, 96])
        tmp = self.down2(x2)
        x3 = self.layer3(tmp) # torch.Size([1, 176, 48, 48])
        tmp = self.down3(x3)
        x4 = self.layer4(tmp) # torch.Size([1, 352, 24, 24])
        tmp = self.down4(x4)
        x5 = self.layer5(tmp) # torch.Size([1, 704, 12, 12])

        # up
        up1 = self.alpha*x4+self.gamma*self.up1(x5) # torch.Size([1, 352, 24, 24])
        up1 = self.layer11(up1)
        up1 = self.channelfix1(up1)

        up2 = self.alpha*x3+self.gamma*self.up2(up1)+self.beta*self.upskip2(x4)
        up2 = self.layer22(up2)
        up2 = self.channelfix2(up2)

        up3 = self.alpha*x2+self.gamma*self.up3(up2)+self.beta*self.upskip3(x3)
        up3 = self.layer33(up3)
        up3 = self.channelfix3(up3)

        up4 = self.alpha*x1+self.gamma*self.up4(up3)+self.beta*self.upskip4(x2) # torch.Size([1, 32, 192, 192])
        up4 = self.layer44(up4)
        up4 = self.channelfix4(up4) # torch.Size([1, 88, 96, 96])

        output = self.output(up4)

        return output


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None, depth=128, norm='bn'):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, depth//4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, int(depth*3/8), kernel_size=1, stride=1, padding=0),
            nn.Conv2d(int(depth*3/8), depth//2, kernel_size=3, stride=1, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, depth//16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(depth//16, depth//8, kernel_size=7, stride=1, padding=3)
        )
        self.bn = nn.BatchNorm2d(in_channel+int(depth*7/8))
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.concat([x1, x, x2, x3], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x



