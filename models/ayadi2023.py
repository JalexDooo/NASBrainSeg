import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d


class ayadi2023(nn.Module): # 
    '''3D

    Paper: https://www.sciencedirect.com/science/article/pii/S1746809422005146
    
    Note: Pointwise Conv == nn.Conv3d() with kernel_size==1

    # flops & params: 666.692G, 20.912M, 16.391844987869263 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(ayadi2023, self).__init__()
        kn = [32, 128, 256, 384]
        # kn = [1, 1, 1, 1]

        self.pre_layer = SingleConvBlock(in_channel, kn[0])
        self.unit1 = nn.Sequential(
            DoubleScaleUnit(kn[0], kn[1]),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )
        self.unit2 = nn.Sequential(
            DoubleScaleUnit(kn[1], kn[2]),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )
        self.mid_layer = SingleConvBlock(kn[2], kn[3])
        self.up_unit1 = UpBlock(kn[3], kn[2])
        self.up_unit2 = UpBlock(kn[2], kn[1])
        self.up_unit3 = UpBlock(kn[1], kn[0])

        self.out_layer = nn.Sequential(
            SingleTransConvBlock(kn[0], kn[0] * 2),
            nn.Conv3d(kn[0] * 2, out_channel, kernel_size=3, stride=1, padding=1)
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        x0 = self.pre_layer(x)

        x1 = self.unit1(x0)

        x2 = self.unit2(x1)

        mid = self.mid_layer(x2)

        x = self.up_unit1(mid, x2)

        x = self.up_unit2(x, x1)

        x = self.up_unit3(x, x0)

        out = self.out_layer(x)

        return out


'''
----------------------epoch 56--------------------
train_loss : 0.006298362467624602
train_dice : 0.8203808439125962
----------------------epoch 57--------------------
train_loss : 0.006016374226836281
train_dice : 0.8454236551921788
----------------------epoch 58--------------------
train_loss : 0.006197508394098436
train_dice : 0.850454255872331
----------------------epoch 59--------------------
train_loss : 0.005685358240287185
train_dice : 0.8277880780209701
'''


class SingleConvBlock(nn.Module):
    """
        正卷积
    """
    def __init__(self, input_data, output_data):
        super(SingleConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleScaleUnit(nn.Module):
    def __init__(self, input_data, output_data):
        super(DoubleScaleUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockWithKernel3(input_data, output_data))
        self.conv.append(ConvBlockWithKernel5(input_data, output_data))



    def forward(self, x):
        x = self.weight1*self.conv[0](x) + self.weight2*self.conv[1](x)

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

def maxpool():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool

class SingleTransConvBlock(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(SingleTransConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=stride, padding=1, output_padding=1,
                               dilation=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlockWithKernel3(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvBlockWithKernel3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(lower=0.125, upper=1.0/3, inplace=True),
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(lower=0.125, upper=1.0/3, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlockWithKernel5(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvBlockWithKernel5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
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



