import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d

def normalization(dim, ntype):
    if ntype == 'bn':
        return nn.BatchNorm3d(dim)
    elif ntype == 'gn':
        return nn.GroupNorm(4, dim)
    elif ntype == 'in':
        return nn.InstanceNorm3d(dim)
    elif ntype == 'ln':
        return nn.LayerNorm(dim)
    elif ntype == 'sync_bn':
        return SynchronizedBatchNorm3d(dim)
    else:
        raise ValueError('normalization type {} is not supported'.format(ntype))


class ConvBlock(nn.Module):
    def __init__(self, in_data, out_data, kernel_size=1, stride=1, groups=1, padding=None, norm=None):
        super(ConvBlock, self).__init__()
        if padding == None:
            padding = (kernel_size-1)//2
        self.bn = normalization(in_data, ntype=norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_data, out_data, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=False)

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return x


class DilatedConvBlock(nn.Module):
    def __init__(self, in_data, out_data, kernel_size=(1, 1, 1), stride=1, groups=1, dilated=(1, 1, 1), norm=None):
        super(DilatedConvBlock, self).__init__()
        padding = tuple(
            [(k-1)//2*d for k, d in zip(kernel_size, dilated)]
        )
        self.bn = normalization(in_data, ntype=norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_data, out_data, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, dilation=dilated, bias=False)

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return x


class AWDBlock(nn.Module):
    def __init__(self, in_data, out_data, groups=1, stride=1, dilated=[1, 2, 3], norm=None):
        super(AWDBlock, self).__init__()
        mid = in_data if in_data <= out_data else out_data
        self.w1 = nn.Parameter(t.ones(1))
        self.w2 = nn.Parameter(t.ones(1))

        self.conv_1 = ConvBlock(in_data, in_data//4, kernel_size=1, stride=1, norm=norm)
        self.conv_2 = ConvBlock(in_data//4, mid, kernel_size=1, stride=1, norm=norm)

        self.d_conv = nn.ModuleList()
        for i in range(3):
            self.d_conv.append(
                DilatedConvBlock(mid, out_data, kernel_size=(3, 3, 3), stride=stride, groups=groups, dilated=(dilated[i], dilated[i], dilated[i]), norm=norm)
            )
        
        self.conv_m = DilatedConvBlock(out_data, out_data, kernel_size=(1, 3, 3), groups=groups, stride=(1, 1, 1), norm=norm)

        if in_data != out_data or stride != 1:
            if stride == 1:
                self.conv_res = ConvBlock(in_data, out_data, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv_res = ConvBlock(in_data, out_data, kernel_size=2, stride=2, padding=0, norm=norm)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.w1*self.d_conv[0](x) + self.w2*self.d_conv[1](x)

        
        x = self.conv_m(x)

        if hasattr(self, 'conv_res'):
            res = self.conv_res(res)

        return x + res


class CBlock(nn.Module):
    def __init__(self, in_data, out_data, groups=1, stride=1, dilated=[1, 2], norm=None):
        super(CBlock, self).__init__()
        mid = in_data if in_data <= out_data else out_data

        self.conv_1 = ConvBlock(in_data, in_data//4, kernel_size=1, stride=1, norm=norm)
        self.conv_2 = ConvBlock(in_data//4, mid, kernel_size=1, stride=1, norm=norm)

        self.d_conv = nn.Sequential(
            DilatedConvBlock(mid, out_data, kernel_size=(3, 3, 3), groups=groups, stride=stride, norm=norm),
            DilatedConvBlock(out_data, out_data, kernel_size=(1, 3, 3), groups=groups, stride=1, norm=norm)
        )

        if in_data != out_data or stride != 1:
            if stride == 1:
                self.conv_res = ConvBlock(in_data, out_data, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv_res = ConvBlock(in_data, out_data, kernel_size=2, stride=2, padding=0, norm=norm)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.d_conv(x)

        if hasattr(self, 'conv_res'):
            res = self.conv_res(res)

        return x + res


class peng2023(nn.Module):
    '''3D Method

    Paper: https://www.sciencedirect.com/science/article/pii/S174680942200790X?via%3Dihub
    
    Note: Open source code.

    # flops & params: 134.98G, 5.27M, 8.3210649490356445 s
    
    '''
    def __init__(self, in_data=4, out_data=4, groups=16, norm='sync_bn', net_arch=None, net_arch_dir=None):
        super(peng2023, self).__init__()
        kn = [32, 64, 128, 256]

        self.encoder1 = ConvBlock(in_data, kn[0], kernel_size=3, stride=2, norm=norm)
        self.encoder2 = nn.Sequential(
            AWDBlock(kn[0], kn[1],groups=groups, stride=2, norm=norm),
            AWDBlock(kn[1], kn[1],groups=groups, stride=1, norm=norm),
            AWDBlock(kn[1], kn[1],groups=groups, stride=1, norm=norm),
        )
        self.encoder3 = nn.Sequential(
            AWDBlock(kn[1], kn[2],groups=groups, stride=2, norm=norm),
            AWDBlock(kn[2], kn[2],groups=groups, stride=1, norm=norm),
            AWDBlock(kn[2], kn[2],groups=groups, stride=1, norm=norm),
        )
        self.encoder4 = nn.Sequential(
            AWDBlock(kn[2], kn[3],groups=groups, stride=2, norm=norm),
            AWDBlock(kn[3], kn[3],groups=groups, stride=1, norm=norm),
            CBlock(kn[3], kn[3],groups=groups, stride=1, norm=norm),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder1 = AWDBlock(kn[3], kn[2], groups=groups, stride=1, norm=norm)
        self.decoder11 = CBlock(kn[2], kn[2], groups=groups, stride=1, norm=norm)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder2 = AWDBlock(kn[2], kn[1], groups=groups, stride=1, norm=norm)
        self.decoder22 = CBlock(kn[1], kn[1], groups=groups, stride=1, norm=norm)


        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder3 = AWDBlock(kn[1], kn[0], groups=groups, stride=1, norm=norm)
        self.decoder33 = CBlock(kn[0], kn[0], groups=groups, stride=1, norm=norm)


        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.decoder4 = ConvBlock(kn[0], out_data, kernel_size=1, stride=1, norm=norm)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                t.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # # 初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = self.up1(x4)
        x = self.decoder1(x)
        x = self.decoder11(x+x3)
        x = self.up2(x)
        x = self.decoder2(x)
        x = self.decoder22(x+x2)
        x = self.up3(x)
        x = self.decoder3(x)
        x = self.decoder33(x+x1)
        x = self.up4(x)
        x = self.decoder4(x)
        x = self.softmax(x)
        return x

