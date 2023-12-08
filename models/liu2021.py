import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BatchNorm3d
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm

from .sync_batchnorm import SynchronizedBatchNorm3d



__all__ = ['CANetOutput', 'CANet']

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
#PREDICT = True
#RESTORE_ID = 10452690098638733452
#RESTORE_EPOCH = 199

#VISUALIZE_PROB_MAP = True

#general settings
SAVE_CHECKPOINTS = True #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "CANet on BraTS19"
EPOCHS = 200
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

#hyperparameters
CHANNELS = 32
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 50 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
RANDOM_CROP = [128, 128, 128]
ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

if not LOG_COMETML:
    experiment = None


class liu2021(nn.Module):
    '''3D Method
    
    Paper: (https://ieeexplore.ieee.org/abstract/document/9378564)

    Note: See https://github.com/ZhihuaLiuEd/canetbrats

    # flops & params: 660.54G, 20.91M
    '''
    def __init__(self, in_channel=4, out_channel=4, norm='bn', net_arch=None, net_arch_dir=None): # num_downsamples is const int 4
        super(liu2021, self).__init__()
        inter_channels = in_channel // 2
        self.conv5a = nn.Sequential(nn.Conv3d(in_channel, inter_channels, 3, padding=1, bias=False),
                                    BatchNorm3d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv3d(in_channel, inter_channels, 3, padding=1, bias=False),
                                    BatchNorm3d(inter_channels),
                                    nn.ReLU())

        self.gcn = nn.Sequential(OrderedDict([("FeatureInteractionGraph%02d" % i,
                                               FeatureInteractionGraph(inter_channels, 30, kernel=1)
                                               ) for i in range(1)]))

        self.dcn = nn.Sequential(OrderedDict([("ConvContextBranch%02d" % i, ConvContextBranch()) for i in range(1)]))

        self.crffusion_1 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_2 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_3 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_4 = CGACRF(inter_channels, inter_channels, inter_channels)
        self.crffusion_5 = CGACRF(inter_channels, inter_channels, inter_channels)

        self.conv51 = normal_conv_blocks(inter_channels, inter_channels)
        self.conv52 = normal_conv_blocks(inter_channels, inter_channels)

        self.upconv1 = normal_conv_blocks(240, 120)
        self.upconv2 = normal_conv_blocks(120, 60)
        self.upconv3 = normal_conv_blocks(120, 60)
        self.upconv4 = normal_conv_blocks(60, 30)
        self.upconv5 = normal_conv_blocks(60, 30)
        self.upconv6 = normal_conv_blocks(30, 30)
        self.final_conv = nn.Conv3d(30, 3, kernel_size=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):

        feat_gcn = self.gcn(x3)
        gcn_conv = self.conv51(feat_gcn)

        feat_dcn = self.dcn(x4)
        fcn_conv = self.conv52(feat_dcn)

        conv_hidden = self.crffusion_1(gcn_conv, fcn_conv)
        conv_hidden = self.crffusion_2(gcn_conv, conv_hidden)
        conv_hidden = self.crffusion_3(gcn_conv, conv_hidden)
        conv_hidden = self.crffusion_4(gcn_conv, conv_hidden)
        conv_hidden = self.crffusion_5(gcn_conv, conv_hidden)

        x = torch.cat([x3, conv_hidden], dim=1)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upsample(x)

        x = torch.cat([x2, x], dim=1)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upsample(x)

        x = torch.cat([x1, x], dim=1)
        x = self.upconv5(x)
        x = self.upconv6(x)

        final_conv_output = self.final_conv(x)
        out = torch.sigmoid(final_conv_output)

        return out


class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        #self.nclass = nclass

        if backbone == 'resnet3d18':
            self.pretrained = resnet3d18()
        elif backbone == 'resnet3d34':
            self.pretrained = resnet3d34()
        elif backbone == 'resnet3d50':
            self.pretrained = resnet3d50()
        elif backbone == 'unet_encoder':
            self.pretrained = unet_encoder()
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        #upsample options
        self._up_kwargs = up_kwargs

    def backbone_forward(self, x):
        conv1 = self.pretrained.dconv_down1(x)
        x1 = self.pretrained.maxpool(conv1)

        conv2 = self.pretrained.dconv_down2(x1)
        x2 = self.pretrained.maxpool(conv2)

        conv3 = self.pretrained.dconv_down3(x2)
        x3 = self.pretrained.maxpool(conv3)

        conv4 = self.pretrained.dconv_down4(x3)

        return conv1, conv2, conv3, conv4


class CANetOutput(Backbone):

    def __init__(self, backbone):
        super(CANetOutput, self).__init__(backbone)
        self.seg_prob = liu2021(240)

    def forward(self, x):

        x1, x2, x3, x4 = self.backbone_forward(x)

        x = self.seg_prob(x1, x2, x3, x4)

        return x





def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    r"""Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
                       bias=None, use_input_stats=None, momentum=None, eps=None):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, int(b * c/group), group, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight=weight, bias=bias,
            training=use_input_stats, momentum=momentum, eps=eps)

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c/group)).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c/group)).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])
    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool3d(inputs, 1).view(inputs.size(0), inputs.size(1), -1)

class BasicBlock(nn.Module):
    """
    ResNet Basic Blocks
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps
    Parameters
    ---------
    block : Block
        class for the residual block. Options are BasicBlockV1, BottlenectV1.
    layers : list of int
        Numbers of layers in each block
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 mode,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    def __init__(self, block, layers, num_classes=3, dilated=False, norm_layer=nn.BatchNorm3d, multi_grid=False, deep_base=True):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()

        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv3d(4, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)

        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)

        self.avgpool = GlobalAvgPool3d()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet3d18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet3d34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet3d50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet3d101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        GroupNorm3d(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        GroupNorm3d(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        GroupNorm3d(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )

class UNetEncoder(nn.Module):
    def __init__(self, n_class=3):
        super(UNetEncoder, self).__init__()
        self.dconv_down1 = double_conv(4, 30)
        self.dconv_down2 = double_conv(30, 60)
        self.dconv_down3 = double_conv(60, 120)
        self.dconv_down4 = double_conv(120, 240)
        self.dconv_down5 = single_conv(240, 480)
        self.dconv_down6 = single_conv(480, 240)

        self.maxpool = nn.MaxPool3d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up4_1 = single_conv(240 + 240, 240)
        self.dconv_up4_2 = single_conv(240, 120)

        self.dconv_up3_1 = single_conv(120 + 120, 120)
        self.dconv_up3_2 = single_conv(120, 60)

        self.dconv_up2_1 = single_conv(60 + 60, 60)
        self.dconv_up2_2 = single_conv(60, 30)

        self.dconv_up1_1 = single_conv(30 + 30, 30)
        self.dconv_up1_2 = single_conv(30, 30)

        self.conv_last = nn.Conv3d(30, n_class, 1)


    def forward(self, x):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        conv6 = self.dconv_down6(conv5)

        x = self.upsample(conv6)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4_1(x)
        x = self.dconv_up4_2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3_1(x)
        x = self.dconv_up3_2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2_1(x)
        x = self.dconv_up2_2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1_1(x)
        x = self.dconv_up1_2(x)

        out = self.conv_last(x)

        return out

def unet_encoder():
    model = UNetEncoder()
    return model

def normal_conv_blocks(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        GroupNorm3d(out_channels, 1),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )

class ConvContextBranch(nn.Module):
    def __init__(self):
        super(ConvContextBranch, self).__init__()
        self.dconv_down1 = normal_conv_blocks(in_channels=240, out_channels=480)
        self.dconv_down2 = normal_conv_blocks(in_channels=480, out_channels=240)
        self.dconv_up1 = normal_conv_blocks(in_channels=480, out_channels=240)
        self.dconv_up2 = normal_conv_blocks(in_channels=240, out_channels=120)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x_mp = self.maxpool(x)
        conv1 = self.dconv_down1(x_mp)
        conv2 = self.dconv_down2(conv1)
        conv2_up = self.upsample(conv2)
        conv2_up_concat = torch.cat([conv2_up, x], dim=1)
        conv3 = self.dconv_up1(conv2_up_concat)
        conv4 = self.dconv_up2(conv3)
        out = self.upsample(conv4)

        return out




class _GroupNorm(_BatchNorm):
    def __init__(self, num_features, num_groups=1, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(int(num_features/num_groups), eps,
                                         momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class GroupNorm2d(_GroupNorm):
    r"""Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    https://arxiv.org/pdf/1803.08494.pdf
    `Group Normalization`_ .
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        num_groups:
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class GroupNorm3d(_GroupNorm):
    """
        Assume the data format is (B, C, D, H, W)
    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

class CGACRF(nn.Module):
    """
    Meanfield updating for the features and the attention for one pair of features.
    bottom_list is a list of observation features derived from the backbone CNN.
    update attention map
    a_s <-- y_s * (K_s conv y_S)
    a_s = b_s conv a_s
    a_s <-- Sigmoid(-(a_s + a_s))
    update the last scale feature map y_S
    y_s <-- K conv y_s
    y_S <-- x_S + (a_s * y_s)
    """

    def __init__(self, bottom_send, bottom_receive, feat_num):
        super(CGACRF, self).__init__()

        self.atten = nn.Conv3d(in_channels=bottom_send + bottom_receive, out_channels=feat_num,
                              kernel_size=3, stride=1, padding=1)
        self.norm_atten = nn.Sigmoid()
        self.message = nn.Conv3d(in_channels=bottom_send, out_channels=feat_num, kernel_size=3,
                                stride=1, padding=1)
        self.scale = nn.Conv3d(in_channels=feat_num, out_channels=bottom_receive, kernel_size=1, bias=True)

    def forward(self, g_s, c_s):
        # x_s -> g_s
        # x_S -> c_s
        # y_s -> g_h
        # y_S -> c_h

        # update attention map
        a_s = torch.cat((g_s, c_s), dim=1)
        a_s = self.atten(a_s)
        a_s = self.norm_atten(a_s)

        # update the last scale feature map y_S
        g_h = self.message(g_s)
        c_h = g_h.mul(a_s)  # production
        c_h = self.scale(c_h) # scale
        c_h = c_s + c_h  # eltwise sum
        return c_h

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.relu(h)
        h = self.conv2(h)
        return h

class FeatureInteractionGraph(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(FeatureInteractionGraph, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel, kernel)
        padding = 1 if kernel == 3 else 0

        # reduce dimension
        self.conv_state = nn.Conv3d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv3d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv3d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=1,
                              groups=1, bias=False)

        self.blocker = BatchNorm3d(num_in)

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # -----------------
        # final
        out = x + self.blocker(self.fc_2(x_state))

        return out

