import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d


class aboelenein2022(nn.Module): # 
    '''2D Method

    Paper: https://www.sciencedirect.com/science/article/pii/S0923596521002733
    
    Note: Gate attention was not detailed.

    # flops & params & time: 2.659G, 2.083M,  0.06959986686706543, 3.340s
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(aboelenein2022, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, 1, 1),
            nn.ReLU(),
        )
        self.b1 = Inception_Res_Block(16, 32)
        self.b2 = Inception_Res_Block(32, 64)
        self.b3 = Inception_Res_Block(64, 128)
        self.b4 = Inception_Res_Block(128, 256)
        self.b5 = Inception_Res_Block(256, 512)

        self.p1 = nn.MaxPool2d(2, 2)
        self.p2 = nn.MaxPool2d(2, 2)
        self.p3 = nn.MaxPool2d(2, 2)
        self.p4 = nn.MaxPool2d(2, 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.b6 = Inception_Res_Block(512, 256)
        self.b7 = Inception_Res_Block(256, 128)
        self.b8 = Inception_Res_Block(128, 64)
        self.b9 = Inception_Res_Block(64, 32)

        self.output = nn.Sequential(
            nn.Conv2d(32, out_channel, 1, 1, 0),
            nn.Softmax(1)
        )

        self.path1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.Conv2d(32, 32, (1,3), 1, (0,1)),
            nn.Conv2d(32, 32, (3,1), 1, (1,0)),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.Conv2d(32, 32, 1, 1, 0),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.Conv2d(64, 64, (1,3), 1, (0,1)),
            nn.Conv2d(64, 64, (3,1), 1, (1,0)),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.Conv2d(64, 64, 1, 1, 0),
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.Conv2d(128, 128, (1,3), 1, (0,1)),
            nn.Conv2d(128, 128, (3,1), 1, (1,0)),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.Conv2d(128, 128, 1, 1, 0),
        )
        self.path4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.Conv2d(256, 256, (1,3), 1, (0,1)),
            nn.Conv2d(256, 256, (3,1), 1, (1,0)),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.Conv2d(256, 256, 1, 1, 0),
        )
    
    def forward(self, x):
        x = self.head(x)
        x1 = self.b1(x)
        x = self.p1(x1)

        x2 = self.b2(x)
        x = self.p2(x2)

        x3 = self.b3(x)
        x = self.p3(x3)

        x4 = self.b4(x)
        x = self.p4(x4)

        x5 = self.b5(x)
        
        x = self.up1(x5)
        x = self.b6(x) + self.path4(x4)

        x = self.up2(x)
        x = self.b7(x) + self.path3(x3)

        x = self.up3(x)
        x = self.b8(x) + self.path2(x2)

        x = self.up4(x)
        x = self.b9(x) + self.path1(x1)

        x = self.output(x)
        return x

class Inception_Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inception_Res_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, 1, 1, 0),
            nn.BatchNorm2d(out_channel//2),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, 1, 1, 0),
            nn.BatchNorm2d(out_channel//2)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, 1, 1, 0),
            nn.Conv2d(out_channel//2, out_channel//2, (1,3), 1, (0,1)),
            nn.Conv2d(out_channel//2, out_channel//2, (3,1), 1, (1,0)),
            nn.BatchNorm2d(out_channel//2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel//2, out_channel//2, 1, 1, 0),
            nn.BatchNorm2d(out_channel//2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2 = self.conv2(x2_1+x2_2)
        x = torch.concat([x1, x2], dim=1)
        return x

