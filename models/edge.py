import torch
import torch.nn as nn

try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

def get_norm_layer(dim, ntype):
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


class Identity(nn.Module):
    def forward(self, x):
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, groups=1, bias=False, norm='bn'):
        super(ConvBlock, self).__init__()
        padding = (kernel_size-1)//2*dilation
        self.conv = nn.Sequential(
            get_norm_layer(in_channel, norm),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias=bias),
        )
    def forward(self, x):
        return self.conv(x)

class HeadConv(nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn'):
        super(HeadConv, self).__init__()
        self.op = nn.Sequential(
            ConvBlock(in_channel, in_channel//4, kernel_size=1, norm=norm),
            ConvBlock(in_channel//4, in_channel, kernel_size=1, norm=norm),
            ConvBlock(in_channel, out_channel, kernel_size=3, norm=norm),
        )

    def forward(self, x):
        return self.op(x)

class MidConv(nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn'):
        super(MidConv, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

        self.conv1 = ConvBlock(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, groups=in_channel//4, norm=norm)
        self.conv2 = ConvBlock(in_channel, out_channel, kernel_size=3, stride=1, dilation=2, groups=in_channel//4, norm=norm)
        self.conv3 = ConvBlock(in_channel, out_channel, kernel_size=3, stride=1, dilation=3, groups=in_channel//4, norm=norm)
    
    def forward(self, x):
        ret = self.weight1*self.conv1(x) + self.weight2*self.conv2(x) + self.weight3*self.conv3(x)
        return ret

class MidAttention(nn.Module):
    def __init__(self, in_channel, out_channel, att_channel, norm='bn'):
        super(MidAttention, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.wx = nn.Sequential(
            nn.Conv3d(in_channel, att_channel, kernel_size=1, bias=True),
            get_norm_layer(att_channel, norm),
        )
        self.wy = nn.Sequential(
            nn.Conv3d(out_channel, att_channel, kernel_size=1, bias=True),
            get_norm_layer(att_channel, norm),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(att_channel, 1, kernel_size=1, bias=True),
            get_norm_layer(1, norm),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv = ConvBlock(in_channel, out_channel, norm=norm)

    def forward(self, x, y):
        wx = self.wx(x)
        wy = self.wy(y)
        psi = self.psi(self.relu(wx+wy))
        ret = self.weight * self.conv(psi*x)
        return ret

class LastConv(nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn'):
        super(LastConv, self).__init__()
        self.op = nn.Sequential(
            ConvBlock(in_channel, out_channel, norm=norm),
        )
    
    def forward(self, x):
        ret = self.op(x)
        return ret

class Edge(nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn', dir=0): # dir -- -1:down, 0:right, 1:up
        super(Edge, self).__init__()
        self.head = HeadConv(in_channel, in_channel, norm=norm)
        self.midConv = MidConv(in_channel, out_channel, norm=norm)
        self.midAtt = MidAttention(in_channel, out_channel, out_channel//2, norm=norm)
        self.last = LastConv(out_channel, out_channel, norm=norm)
        self.skip_connection = ConvBlock(in_channel, out_channel, norm=norm)

        if dir == -1:
            self.dir = nn.MaxPool3d(2, 2)
        elif dir == 0:
            self.dir = Identity()
        elif dir == 1:
            self.dir = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    
    def forward(self, x):
        res = self.skip_connection(x)
        head = self.head(x)
        midConv = self.midConv(head)
        # midAtt = self.midAtt(head, midConv)
        last = self.last(midConv) # +midAtt)
        output = last + res
        ret = self.dir(output)

        return ret

if __name__ == '__main__':
    import numpy as np
    cell = Edge(16, 32)
    a = torch.Tensor(np.random.randn(2, 16, 32, 32, 32))
    out = cell(a)
    print(out.shape)
