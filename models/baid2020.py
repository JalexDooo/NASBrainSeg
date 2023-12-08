import os
import torch
import torch.nn as nn
from models.base_model import BaseModel, init_net, get_norm_layer, GeneralizedDiceLoss, expand_target
from monai.losses import DiceCELoss, DiceLoss


class baid2020(nn.Module):
    '''3D Method

    Paper: https://www.frontiersin.org/articles/10.3389/fncom.2020.00010/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Computational_Neuroscience&id=495068
    
    Note:

    # flops & params: 968.850G, 8.998M, 106.92006921768188 s
    
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(baid2020, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.pool3 = nn.MaxPool3d(2, 2)

        self.deep = nn.Sequential(
            nn.Conv3d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(384, 384, kernel_size=2, stride=2, output_padding=0),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(576, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2, output_padding=0),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(96, 96, kernel_size=2, stride=2, output_padding=0),
        )
        self.output = nn.Sequential(
            nn.Conv3d(144, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(48, 4, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x):
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        x3 = self.conv3(x)
        x = self.pool3(x3)
        x = self.deep(x)
        x = torch.concat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.concat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.concat([x, x1], dim=1)
        x = self.output(x)
        return x