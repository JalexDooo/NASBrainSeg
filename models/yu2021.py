import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d

class LuT(nn.Module):
    # a = t.tensor(np.random.randn(8, 4, 128, 128, 128)).float()
    def __init__(self, in_data=4, out_data=4):
        super(LuT, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_data, 16, kernel_size=4, stride=4),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=4, stride=4),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, stride=4),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dp = nn.Dropout3d(p=0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, out_data),
            nn.BatchNorm1d(out_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        bz = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dp(x)
        x = x.view(bz, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x # [bz, (a1, a3, s1, s2)]


class ThreeSegWise(nn.Module):
    def __init__(self, in_data=4, out_data=4):
        super(ThreeSegWise, self).__init__()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, images, eps=1e-8):
        for i in range(x.size(0)):
            a1 = x[i, 0]
            a3 = x[i, 1]
            s1 = x[i, 2]
            s2 = x[i, 3]
            image = images[i, :]
            seg1 = (image*(image<s1)*(image>=0.0)) * a1 + eps
            seg2 = (image*(image<=(s1+s2))*(image>=s1)) * (a3*(s1+s2-1-a1*s1+1)+eps)/(s2+eps) + (((a1-a3)*(s1+s2)*s1 + (a3-1)*s1)+eps)/(s2+eps)
            seg3 = (image*(image<=1)*(image>(s1+s2))) * a3 - a3+1 + eps
            if i==0:
                x__ = (seg1+seg2+seg3).unsqueeze(dim=0)
            else:
                x__ = t.cat([x__, (seg1+seg2+seg3).unsqueeze(dim=0)], dim=0)
        x__ = self.relu(x__ + images)
        return x__


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in,norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFunit, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid,num_out,kernel_size=(3,3,3),stride=stride,g=g,d=(d[0],d[0],d[0]),norm=norm) # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(3,3,1),stride=1,g=g,d=(d[1],d[1],1),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x

        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut

class DMFUnit(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(DMFUnit, self).__init__()
        self.weight1 = nn.Parameter(t.ones(1))
        self.weight2 = nn.Parameter(t.ones(1))
        self.weight3 = nn.Parameter(t.ones(1))

        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2,3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_mid,num_out, kernel_size=(3, 3, 3), stride=stride, g=g, d=(dilation[i],dilation[i], dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)


    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.weight1*self.conv3x3x3_m1[0](x2) + self.weight2*self.conv3x3x3_m1[1](x2) + self.weight3*self.conv3x3x3_m1[2](x2)
        x4 = self.conv3x3x3_m2(x3)
        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        return x4 + shortcut


class MFNet(nn.Module): #
    # [96]   Flops:  13.361G  &  Params: 1.81M
    # [112]  Flops:  16.759G  &  Params: 2.46M
    # [128]  Flops:  20.611G  &  Params: 3.19M
    def __init__(self, c=4,n=32,channels=128,groups = 16,norm='bn', num_classes=4):
        super(MFNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=2, bias=False)# H//2
        self.encoder_block2 = nn.Sequential(
            MFunit(n, channels, g=groups, stride=2, norm=norm),# H//4 down
            MFunit(channels, channels, g=groups, stride=1, norm=norm),
            MFunit(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MFunit(channels, channels*2, g=groups, stride=2, norm=norm), # H//8
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(# H//8,channels*4
            MFunit(channels*2, channels*3, g=groups, stride=2, norm=norm), # H//16
            MFunit(channels*3, channels*3, g=groups, stride=1, norm=norm),
            MFunit(channels*3, channels*2, g=groups, stride=1, norm=norm),
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.decoder_block1 = MFunit(channels*2+channels*2, channels*2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.decoder_block2 = MFunit(channels*2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                t.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm): #or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)# H//2 down
        x2 = self.encoder_block2(x1)# H//4 down
        x3 = self.encoder_block3(x2)# H//8 down
        x4 = self.encoder_block4(x3) # H//16
        # Decoder
        y1 = self.upsample1(x4)# H//8
        y1 = t.cat([x3,y1],dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)# H//4
        y2 = t.cat([x2,y2],dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)# H//2
        y3 = t.cat([x1,y3],dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self,'softmax'):
            y4 = self.softmax(y4)
        return y4


class DMFNet(MFNet): # softmax
    # [128]  Flops:  27.045G  &  Params: 3.88M
    def __init__(self, c=4,n=32,channels=128, groups=16,norm='bn', num_classes=4):
        super(DMFNet, self).__init__(c,n,channels,groups, norm, num_classes)

        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm,dilation=[1,2,3]),# H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3]), # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels*2, g=groups, stride=2, norm=norm,dilation=[1,2,3]), # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3]),# Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3])
        )

class yu2021(nn.Module): # SA_LuT_Nets
    '''3D Method

    Paper: https://ieeexplore.ieee.org/abstract/document/9345772
    
    Note:

    # flops & params: 22.263G, 3.851M for [192, 192, 48] without self.lut or self.segwise3.
    # flops & params: 26.539G, 4.094M for [128, 128, 128], 7.380129098892212 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(yu2021, self).__init__()
        self.lut = LuT()
        self.segwise3 = ThreeSegWise()
        self.dmf = DMFNet(c=4, groups=16, norm='sync_bn', num_classes=4)
    
    def forward(self, x):
        x_ = self.lut(x)
        x_ = self.segwise3(x_, x)
        x_ = self.dmf(x_)
        return x_
