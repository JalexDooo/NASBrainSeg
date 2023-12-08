import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d

class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(),
            nn.Conv3d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x) + x
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(),
            nn.Conv3d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AttentionBlock, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention(in_channel)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
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

class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttention, self).__init__()
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

class liu2023(nn.Module): # 
    '''3D Method

    Paper: https://www.sciencedirect.com/science/article/pii/S1319157823000794
    
    Note: N=48, Dilation is not detailed.

    # flops & params: 2.511T, 73.679M, 35.34598183631897 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(liu2023, self).__init__()
        kn = [48, 48*2, 48*4, 48*8]
        self.head = nn.Conv3d(in_channel, kn[0], 1, 1, 0)
        self.conv1 = nn.Sequential(
            EncoderBlock(kn[0], kn[0]),
            nn.Conv3d(kn[0], kn[0], 3, 1, 1),
            EncoderBlock(kn[0], kn[0]),
            nn.Conv3d(kn[0], kn[0], 3, 1, 1),
            AttentionBlock(kn[0], kn[0])
        )
        self.conv2 = nn.Sequential(
            EncoderBlock(kn[1], kn[1]),
            nn.Conv3d(kn[1], kn[1], 3, 1, 1),
            EncoderBlock(kn[1], kn[1]),
            nn.Conv3d(kn[1], kn[1], 3, 1, 1),
            AttentionBlock(kn[1], kn[1])
        )
        self.conv3 = nn.Sequential(
            EncoderBlock(kn[2], kn[2]),
            nn.Conv3d(kn[2], kn[2], 3, 1, 1),
            EncoderBlock(kn[2], kn[2]),
            nn.Conv3d(kn[2], kn[2], 3, 1, 1),
            AttentionBlock(kn[2], kn[2])
        )
        self.conv4 = nn.Sequential(
            EncoderBlock(kn[3], kn[3]),
            nn.Conv3d(kn[3], kn[3], 3, 1, 1),
            EncoderBlock(kn[3], kn[3]),
            nn.Conv3d(kn[3], kn[3], 3, 1, 1),
            EncoderBlock(kn[3], kn[3]),
            nn.Conv3d(kn[3], kn[3], 3, 1, 2, dilation=2),
            EncoderBlock(kn[3], kn[3]),
            nn.Conv3d(kn[3], kn[3], 3, 1, 2, dilation=2),
            EncoderBlock(kn[3], kn[3]),
        )
        self.down1 = nn.Conv3d(kn[0], kn[1], 3, 2, 1)
        self.down2 = nn.Conv3d(kn[1], kn[2], 3, 2, 1)
        self.down3 = nn.Conv3d(kn[2], kn[3], 3, 2, 1)
        self.up1 = nn.ConvTranspose3d(kn[3], kn[2], kernel_size=2, stride=2, output_padding=0)
        self.up2 = nn.ConvTranspose3d(kn[2], kn[1], kernel_size=2, stride=2, output_padding=0)
        self.up3 = nn.ConvTranspose3d(kn[1], kn[0], kernel_size=2, stride=2, output_padding=0)

        self.upconv1 = nn.Sequential(
            DecoderBlock(kn[2], kn[2]),
            nn.Conv3d(kn[2], kn[2], 3, 1, 1),
            DecoderBlock(kn[2], kn[2]),
            nn.Conv3d(kn[2], kn[2], 3, 1, 1),
            DecoderBlock(kn[2], kn[2]),
        )
        self.upconv2 = nn.Sequential(
            DecoderBlock(kn[1], kn[1]),
            nn.Conv3d(kn[1], kn[1], 3, 1, 1),
            DecoderBlock(kn[1], kn[1]),
            nn.Conv3d(kn[1], kn[1], 3, 1, 1),
            DecoderBlock(kn[1], kn[1]),
        )
        self.upconv3 = nn.Sequential(
            DecoderBlock(kn[0], kn[0]),
            nn.Conv3d(kn[0], kn[0], 3, 1, 1),
            DecoderBlock(kn[0], kn[0]),
            nn.Conv3d(kn[0], kn[0], 3, 1, 1),
            DecoderBlock(kn[0], kn[0]),
        )
        self.output = nn.Conv3d(kn[0], 4, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x1 = self.conv1(x)
        x = self.down1(x1)

        x2 = self.conv2(x)
        x = self.down2(x2)

        x3 = self.conv3(x)
        x = self.down3(x3)

        x = self.conv4(x)

        x = self.up1(x)
        x = self.upconv1(x+x3)

        x = self.up2(x)
        x = self.upconv2(x+x2)

        x = self.up3(x)
        x = self.upconv3(x+x1)
        x = self.output(x)

        return x