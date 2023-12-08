import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d


class raza2023(nn.Module): # 
    '''3D Method

    Paper: https://www.sciencedirect.com/science/article/pii/S1746809422003809
    
    Note: Not find open source Code in https://github.com/rehanrazaa/dResU-Net_Deep_Residual_U-Net_Brain_Tumor_Segmentation

    # flops & params: 1.076T, 26.181M, 34.923763036727905 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(raza2023, self).__init__()
        kn = [32, 64, 128, 256]

        self.in_model = NewConvDown(in_channel, kn[0], stride=1)

        self.layer1 = nn.Sequential(
            NewResBlock(kn[0], kn[0]),
            NewResBlock(kn[0], kn[0]),
            NewConvDown(kn[0], kn[1], stride=2)
        )
        self.layer2 = nn.Sequential(
            NewResBlock(kn[1], kn[1]),
            NewResBlock(kn[1], kn[1]),
            NewConvDown(kn[1], kn[2], stride=2)
        )
        self.layer3 = nn.Sequential(
            NewResBlock(kn[2], kn[2]),
            NewResBlock(kn[2], kn[2]),
            NewConvDown(kn[2], kn[3], stride=2)
        )

        self.bottom = nn.Sequential(
            NewResBlock(kn[3], kn[3]),
            NewResBlock(kn[3], kn[3]),
            NewResBlock(kn[3], kn[3]),
            NewConvUp(kn[3], kn[2], stride=2)
        )

        self.llayer3 = nn.Sequential(
            NewResBlock(kn[2]*2, kn[2]),
            NewResBlock(kn[2], kn[2]),
            NewConvUp(kn[2], kn[1], stride=2)
        )

        self.llayer2 = nn.Sequential(
            NewResBlock(kn[1]*2, kn[1]),
            NewResBlock(kn[1], kn[1]),
            NewConvUp(kn[1], kn[0], stride=2)
        )

        self.out = OutConv(kn[0]*2, out_channel)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ind):
        x_ = self.in_model(ind)
        x1 = self.layer1(x_)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        bottom = self.bottom(x3)

        # print('shape: bottom: ', bottom.shape)
        # print('shape: x3: ', x3.shape)
        # print('shape: x2: ', x2.shape)
        # print('shape: x1: ', x1.shape)

        x = torch.cat([bottom, x2], dim=1)
        x = self.llayer3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.llayer2(x)
        x = torch.cat([x, x_], dim=1)

        out = self.out(x)
        return out



class NewResBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(NewResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.BatchNorm3d(input_data),
            nn.RReLU(lower=0.125, upper=1.0/3, inplace=True),
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True),
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm3d(input_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True),
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        res = self.resblock(x)
        x = self.conv(x)
        x += res
        return x



class NewConvDown(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(NewConvDown, self).__init__()
        self.conv = nn.Conv3d(input_data, output_data, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class NewConvUp(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(NewConvUp, self).__init__()
        self.conv = nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=stride, padding=1, output_padding=1, dilation=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, input_data, output_data):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(input_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True),
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x