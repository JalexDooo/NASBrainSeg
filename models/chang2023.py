import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d


class chang2023(nn.Module): # 
    '''3D Method

    Paper: https://www.sciencedirect.com/science/article/pii/S1746809422005146
    
    Note: Pointwise Conv == nn.Conv3d() with kernel_size==1

    # flops & params: 144.880G, 4.866M, 9.3448309898376465 s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(chang2023, self).__init__()
        kn = [32, 64, 128, 256]
        self.head = nn.Conv3d(in_channel, kn[0], 3, 1, 1)

        self.down1 = nn.Conv3d(kn[0], kn[1], 3, 2, 1)
        self.down2 = nn.Conv3d(kn[1], kn[2], 3, 2, 1)
        self.down3 = nn.Conv3d(kn[2], kn[3], 3, 2, 1)

        self.dp1 = DPModule(kn[1])
        self.dp2 = DPModule(kn[2])
        self.dp3 = DPModule(kn[3])

        self.encoder1 = MAFModule(kn[1], kn[1])
        self.encoder2 = MAFModule(kn[2], kn[2])
        self.encoder3 = MAFModule(kn[3], kn[3])

        self.up1 = nn.ConvTranspose3d(kn[3], kn[2], kernel_size=2, stride=2, output_padding=0)
        self.up2 = nn.ConvTranspose3d(kn[2], kn[1], kernel_size=2, stride=2, output_padding=0)
        self.up3 = nn.ConvTranspose3d(kn[1], kn[0], kernel_size=2, stride=2, output_padding=0)

        self.decoder1 = nn.Sequential(
            nn.Conv3d(kn[2]+kn[2], kn[2], 1, 1, 0),
            nn.BatchNorm3d(kn[2]),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(kn[1]+kn[1], kn[1], 1, 1, 0),
            nn.BatchNorm3d(kn[1]),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(kn[0]+kn[0], kn[0], 1, 1, 0),
            nn.BatchNorm3d(kn[0]),
            nn.ReLU(True)
        )
        self.output = nn.Conv3d(kn[0], out_channel, 1, 1, 0)
    
    def forward(self, x):
        x1 = self.head(x)
        x = self.down1(x1)

        dp1_s1, dp1_s2 = self.dp1(x)
        x2 = self.encoder1(dp1_s1, dp1_s2)
        x = self.down2(x2)

        dp2_s1, dp2_s2 = self.dp2(x)
        x3 = self.encoder2(dp2_s1, dp2_s2)
        x = self.down3(x3)

        dp3_s1, dp3_s2 = self.dp3(x)
        x4 = self.encoder3(dp3_s1, dp3_s2)

        x = self.up1(x4)
        x = torch.concat([x, x3], dim=1)
        x = self.decoder1(x)

        x = self.up2(x)
        x = torch.concat([x, x2], dim=1)
        x = self.decoder2(x)

        x = self.up3(x)
        x = torch.concat([x, x1], dim=1)
        x = self.decoder3(x)

        x = self.output(x)

        return x

class DPModule(nn.Module):
    '''
    Has BatchNorm or ReLU? This paper was not provided.
    '''
    def __init__(self, in_channel):
        super(DPModule, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//8, 1, 1, 0),
            nn.BatchNorm3d(in_channel//8),
            nn.ReLU(True),
            nn.Conv3d(in_channel//8, in_channel//4, 3, 1, 1),
            nn.BatchNorm3d(in_channel//4),
            nn.ReLU(True),
            nn.Conv3d(in_channel//4, in_channel//4, 3, 1, 1),
            nn.BatchNorm3d(in_channel//4),
            nn.ReLU(True),
            nn.Conv3d(in_channel//4, in_channel, 1, 1, 0),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(True),
        )
        self.s2 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel//4, 1, 1, 0),
            nn.BatchNorm3d(in_channel//4),
            nn.ReLU(True),
            nn.Conv3d(in_channel//4, in_channel, 5, 1, 2),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(True),
            nn.Conv3d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(x)+x
        return s1, s2

class MAFModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MAFModule, self).__init__()
        self.conv1 = nn.Sequential( # PW, PW
            nn.Conv3d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(True),
            nn.Conv3d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential( # GAP, PW. But GAP function -> [batch, channel, 1], how to add self.conv1 (5D array).
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channel, in_channel, 1, 1, 0),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(True),
        )
        self.sig = nn.Sigmoid()

    def forward(self, s1, s2):
        s = s1+s2
        x1 = self.conv1(s)
        x2 = self.conv2(s)
        x = F.sigmoid(x1+x2)
        s1 = s1*x
        s2 = s2*x
        z = s1+s2

        return z


