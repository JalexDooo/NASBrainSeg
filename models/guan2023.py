import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        padding = kernel_size//2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.gpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.gpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class AGBlock(nn.Module):
    def __init__(self):
        super(AGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, 7, 1, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg1 = torch.mean(x, dim=1, keepdim=True) # average pool in channel
        max1, _ = torch.max(x, dim=1, keepdim=True) # max pool in channel
        cat = torch.cat([avg1, max1], dim=1)
        cat = self.conv(cat)
        x = x*cat
        return x


class guan2023(nn.Module): # 
    '''3D Method

    Paper: https://www.researchgate.net/figure/The-overall-architecture-of-the-proposed-3D-AGSE-VNet_fig1_353478396
    
    Note: AGBlock is not detail.

    # flops & params: 398.160G, 31.780M, 6.556315898895264 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(guan2023, self).__init__()
        kn = [32, 64, 128, 256, 512]
        self.head = nn.Conv3d(in_channel, 32, 1, 1, 0)
        self.conv1 = nn.Sequential(
            ConvBlock(kn[0], kn[0]),
            ConvBlock(kn[0], kn[0]),
            SEBlock(kn[0])
        )
        self.conv2 = nn.Sequential(
            ConvBlock(kn[1], kn[1]),
            ConvBlock(kn[1], kn[1]),
            SEBlock(kn[1])
        )
        self.conv3 = nn.Sequential(
            ConvBlock(kn[2], kn[2]),
            ConvBlock(kn[2], kn[2]),
            ConvBlock(kn[2], kn[2]),
            SEBlock(kn[2])
        )
        self.conv4 = nn.Sequential(
            ConvBlock(kn[3], kn[3]),
            ConvBlock(kn[3], kn[3]),
            ConvBlock(kn[3], kn[3]),
            SEBlock(kn[3])
        )

        self.down1 = nn.Conv3d(kn[0], kn[1], 3, 2, 1)
        self.down2 = nn.Conv3d(kn[1], kn[2], 3, 2, 1)
        self.down3 = nn.Conv3d(kn[2], kn[3], 3, 2, 1)
        self.down4 = nn.Conv3d(kn[3], kn[4], 3, 2, 1)

        self.deep = nn.Sequential(
            ConvBlock(kn[4], kn[4]),
            ConvBlock(kn[4], kn[4]),
            SEBlock(kn[4])
        )

        self.up1 = nn.ConvTranspose3d(kn[4], kn[3], kernel_size=2, stride=2, output_padding=0)
        self.up2 = nn.ConvTranspose3d(kn[3], kn[2], kernel_size=2, stride=2, output_padding=0)
        self.up3 = nn.ConvTranspose3d(kn[2], kn[1], kernel_size=2, stride=2, output_padding=0)
        self.up4 = nn.ConvTranspose3d(kn[1], kn[0], kernel_size=2, stride=2, output_padding=0)

        self.decode1 = nn.Sequential(
            ConvBlock(kn[3], kn[3]),
            ConvBlock(kn[3], kn[3]),
            AGBlock()
        )
        self.decode2 = nn.Sequential(
            ConvBlock(kn[2], kn[2]),
            ConvBlock(kn[2], kn[2]),
            AGBlock()
        )
        self.decode3 = nn.Sequential(
            ConvBlock(kn[1], kn[1]),
            AGBlock()
        )
        self.decode4 = nn.Sequential(
            ConvBlock(kn[0], kn[0]),
            AGBlock()
        )
        self.output = nn.Sequential(
            nn.Conv3d(kn[0], out_channel, 1, 1, 0),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.head(x)
        x1 = self.conv1(x)
        x = self.down1(x1)

        x2 = self.conv2(x)
        x = self.down2(x2)

        x3 = self.conv3(x)
        x = self.down3(x3)

        x4 = self.conv4(x)
        x = self.down4(x4)
        x = self.deep(x)

        x = self.up1(x)
        x = self.decode1(x+x4)

        x = self.up2(x)
        x = self.decode2(x+x3)

        x = self.up3(x)
        x = self.decode3(x+x2)

        x = self.up4(x)
        x = self.decode4(x+x1)
        x = self.output(x)

        # print('xxx: ', x.shape)

        return x