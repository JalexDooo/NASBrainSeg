import os
import torch
import torch.nn as nn
from models.base_model import BaseModel, init_net, get_norm_layer, GeneralizedDiceLoss
from models.edge import Edge, ConvBlock

class SuperNet(nn.Module):
    def __init__(self, in_channel=4, out_channel=4, norm='bn', num_layers=8, num_downsamples=4): # num_downsamples is const int 4
        super(SuperNet, self).__init__()
        self._num_layers = num_layers
        self._num_downsamples = num_downsamples
        self._channels = {0:16, 1:32, 2:64, 3:128, 4:256} # downsample:channel
        self._betas = nn.Parameter(torch.ones((self._num_layers, self._num_downsamples)))
        self._edges = nn.ModuleList()
        self.head = ConvBlock(in_channel, 16, stride=2, norm=norm)

        for i in range(self._num_layers):
            for j in range(i+1):
                if j >= self._num_downsamples:
                    break
                if j == 0:
                    self._edges += [Edge(self._channels[j], self._channels[j], norm=norm, dir=0)]
                    self._edges += [Edge(self._channels[j], self._channels[j+1], norm=norm, dir=-1)]
                elif j == self._num_downsamples-1:
                    self._edges += [Edge(self._channels[j], self._channels[j-1], norm=norm, dir=1)]
                    self._edges += [Edge(self._channels[j], self._channels[j], norm=norm, dir=0)]
                else:
                    self._edges += [Edge(self._channels[j], self._channels[j-1], norm=norm, dir=1)]
                    self._edges += [Edge(self._channels[j], self._channels[j], norm=norm, dir=0)]
                    self._edges += [Edge(self._channels[j], self._channels[j+1], norm=norm, dir=-1)]
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(self._channels[0], 16),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(self._channels[1], self._channels[0]),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(self._channels[0], 16),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(self._channels[2], self._channels[1]),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(self._channels[1], self._channels[0]),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(self._channels[0], 16),
        )
        # self.up4 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        #     ConvBlock(self._channels[3], self._channels[2]),
        #     nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        #     ConvBlock(self._channels[2], self._channels[1]),
        #     nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        #     ConvBlock(self._channels[1], self._channels[0]),
        #     nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
        #     ConvBlock(self._channels[0], 16),
        # )
        self.last = nn.Sequential(
            ConvBlock(16, out_channel),
            nn.Softmax(dim=1),
        )

    def forward(self, x): # betas
        features_channel = [[] for _ in range(self._num_downsamples)]
        
        x = self.head(x)
        features_channel[0].append(x)
        k = 0
        for i in range(self._num_layers):
            cc = [0 for _ in range(self._num_downsamples)]
            dd = [0 for _ in range(self._num_downsamples)]
            for j in range(i+1):
                if j >= self._num_downsamples:
                    break
                if j == 0:
                    fet0 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    fet1 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    # update features
                    cc[j] += fet0
                    cc[j+1] += fet1
                    dd[j] += 1
                    dd[j+1] += 1
                elif j == self._num_downsamples-1:
                    fet0 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    fet1 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    # update features
                    cc[j-1] += fet0
                    cc[j] += fet1
                    dd[j-1] += 1
                    dd[j] += 1
                else:
                    fet0 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    fet1 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    fet2 = self._edges[k](features_channel[j][-1])
                    k = k+1
                    # update features
                    cc[j-1] += fet0
                    cc[j] += fet1
                    cc[j+1] += fet2
                    dd[j-1] += 1
                    dd[j] += 1
                    dd[j+1] += 1
            for j in range(self._num_downsamples):
                features_channel[j].append(cc[j]/(max(dd[j], 1))*self._betas[i][j])

        # for i in range(self._num_downsamples):
        #     print('i: ', i, features_channel[i][-1].shape, len(features_channel[i]))
        
        # upsample and skip connection
        o1 = self.up1(features_channel[0][-1] + features_channel[0][0])
        o2 = self.up2(features_channel[1][-1] + features_channel[1][0])
        o3 = self.up3(features_channel[2][-1] + features_channel[2][0])
        # o4 = self.up4(features_channel[3][-1] + features_channel[3][0])
        ret = self.last((o1+o2+o3))#+o4)
        return ret

def define_net(in_channel, out_channel, norm='bn', num_layers=8, num_downsamples=4, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = SuperNet(in_channel, out_channel, norm, num_layers, num_downsamples)
    return init_net(net, init_type, init_gain, gpu_ids)

class SearchNet(BaseModel):
    def __init__(self, cfg):
        BaseModel.__init__(self, cfg)
        self.loss_names = ['gdl']
        self.visual_names = ['img', 'label', 'pred']

        self.model_names = ['Search']

        self.netSearch = define_net(1, 14, 'bn', 8, 3, init_type='kaiming', gpu_ids=self.gpu_ids)

        if self.isTrain:
            # self.criterion = GeneralizedDiceLoss
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.netSearch.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)
    
    def betas(self):
        return self.netSearch._betas
                    
    def set_input(self, input):
        self.img = input['img'].to(self.device)
        self.label = input['label'].to(self.device)
    
    def forward(self):
        self.pred = self.netSearch(self.img)

    def backward(self):
        self.loss_gdl = self.criterion(self.pred, self.label.long())
        self.loss_gdl.backward()
    
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()



