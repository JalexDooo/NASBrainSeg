import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d


class ghaffari2022(nn.Module): # 
    '''

    Paper: Automated post-operative brain tumour segmentation: A deep learning model based on transfer learning from pre-operative images
    
    Note: UNet

    # flops & params: 150.960G, 6.478M, 4.100227117538452 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(ghaffari2022, self).__init__()
        degree = 16

        drop = []
        for i in range(5):
            drop.append((2 ** i) * degree)
        
        self.downLayer1 = ConvBlock(in_channel, drop[0])
        self.downLayer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlock(drop[0], drop[1])
        )
        self.downLayer3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlock(drop[1], drop[2])
        )
        self.downLayer4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlock(drop[2], drop[3])
        )
        self.bottomLayer = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlock(drop[3], drop[4])
        )

        self.upLayer1 = UpBlock(drop[4], drop[3])
        self.upLayer2 = UpBlock(drop[3], drop[2])
        self.upLayer3 = UpBlock(drop[2], drop[1])
        self.upLayer4 = UpBlock(drop[1], drop[0])

        self.outLayer = nn.Conv3d(drop[0], out_channel, kernel_size=3, stride=1, padding=1)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.downLayer1(x)
        x2 = self.downLayer2(x1)
        x3 = self.downLayer3(x2)
        x4 = self.downLayer4(x3)

        bottom = self.bottomLayer(x4)

        x = self.upLayer1(bottom, x4)
        x = self.upLayer2(x, x3)
        x = self.upLayer3(x, x2)
        x = self.upLayer4(x, x1)
        x = self.outLayer(x)
        return x


class ConvBlock(nn.Module):
    """
        正卷积
    """
    def __init__(self, input_data, output_data):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(UpBlock, self).__init__()
        self.up = ConvTransBlock(input_data, output_data)
        self.down = ConvBlock(2*output_data, output_data)
    
    def forward(self, x, down_features):
        x = self.up(x)
        x = torch.cat([x, down_features], dim=1) # 横向拼接
        x = self.down(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvTransBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        return x
    
