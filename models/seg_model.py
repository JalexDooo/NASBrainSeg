import os
import torch
import torch.nn as nn
from models.base_model import BaseModel, init_net, get_norm_layer, GeneralizedDiceLoss, expand_target
from models.edge import Edge, ConvBlock
from monai.losses import DiceCELoss, DiceLoss

# path = [1, 1, 0, 0, 1, 1, 2, 2]
# path_dir = [-1, 0, 1, 0, -1, 0, -1, 0] # -1:down, 0:same, 1:up
class SuperNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(SuperNet, self).__init__()
        self._net_arch = net_arch
        self._net_arch_dir = net_arch_dir
        self._channels = {0:16, 1:32, 2:64, 3:128, 4:256} # downsample:channel

        self._edges = nn.ModuleList()
        self.head = ConvBlock(in_channel, self._channels[0], stride=2, norm=norm)

        for i in range(len(self._net_arch)):
            if not i:
                self._edges += [Edge(self._channels[0], self._channels[self._net_arch[i]], norm=norm, dir=self._net_arch_dir[0])]
                continue
            if self._net_arch_dir[i] == 0:
                self._edges += [Edge(self._channels[self._net_arch[i]], self._channels[self._net_arch[i]], norm=norm, dir=0)]
            elif self._net_arch_dir[i] == 1:
                self._edges += [Edge(self._channels[self._net_arch[i-1]], self._channels[self._net_arch[i]], norm=norm, dir=1)]
            elif self._net_arch_dir[i] == -1:
                self._edges += [Edge(self._channels[self._net_arch[i-1]], self._channels[self._net_arch[i]], norm=norm, dir=-1)]
        
        self._summ = sum(self._net_arch_dir)
        if self._summ == 0:
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ConvBlock(self._channels[0], self._channels[0], norm=norm),
            )
        elif self._summ == -1:
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ConvBlock(self._channels[1], self._channels[0], norm=norm),
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ConvBlock(self._channels[0], self._channels[0], norm=norm),
            )
        elif self._summ == -2:
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ConvBlock(self._channels[2], self._channels[1], norm=norm),
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ConvBlock(self._channels[1], self._channels[0], norm=norm),
            )
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                ConvBlock(self._channels[0], self._channels[0], norm=norm),
            )

        self.last = nn.Sequential(
            ConvBlock(self._channels[0], out_channel, norm=norm),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        features_channel = [[] for _ in range(max(self._net_arch) +1)]
        
        x = self.head(x)
        features_channel[0].append(x)

        for i in range(len(self._net_arch)):
            x = self._edges[i](x)
            if len(features_channel[self._net_arch[i]]) == 0:
                features_channel[self._net_arch[i]].append(x)

        if self._summ == 0:
            x = self.up1(x + features_channel[0][0])
        elif self._summ == -1:
            x = self.up1(x + features_channel[1][0])
            x = self.up2(x + features_channel[0][0])
        elif self._summ == -2:
            # upsample and skip connection
            x = self.up1(x + features_channel[2][0])
            x = self.up2(x + features_channel[1][0])
            x = self.up3(x + features_channel[0][0])
        ret = self.last(x)
        return ret

def define_net(in_channel, out_channel, norm='bn', net_arch=None, net_arch_dir=None, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = SuperNet(in_channel, out_channel, norm, net_arch, net_arch_dir)
    return init_net(net, init_type, init_gain, gpu_ids)

class SegNet(BaseModel):
    def __init__(self, cfg):
        BaseModel.__init__(self, cfg)
        self.loss_names = ['gdl']
        self.visual_names = ['img', 'label', 'pred']

        self.model_names = ['Seg']
        self._net_arch = cfg.net_arch
        self._net_arch_dir = cfg.net_arch_dir

        self.netSeg = define_net(1, 14, 'sync_bn', self._net_arch, self._net_arch_dir, init_type='kaiming', gpu_ids=self.gpu_ids)

        if self.isTrain:
            # self.criterion = GeneralizedDiceLoss
            # self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True)
            self.optimizer = torch.optim.Adam(self.netSeg.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)
                    
    def set_input(self, input):
        self.img = input['img'].to(self.device)
        self.label = input['label'].to(self.device)
        self.label = expand_target(self.label, n_class=14, mode='sigmoid')
        # print('shape:,,,,', self.img.shape, self.label.shape)
    
    def forward(self):
        self.pred = self.netSeg(self.img)

    def backward(self):
        self.loss_gdl = self.criterion(self.pred, self.label.long())
        self.loss_gdl.backward()
    
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()



