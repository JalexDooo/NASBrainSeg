import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm import SynchronizedBatchNorm3d
import torch
import torch.nn as nn
import torch.nn.functional as F


norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        self.activation = activation
        self.leaky = leaky
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if self.leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        # instantiate layers
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(8, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ScaleUpsample(nn.Module):
    def __init__(self, use_deconv=False, num_channels=None, scale_factor=None, mode='trilinear', align_corners=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        if use_deconv:
            self.trans_conv = nn.ConvTranspose3d(num_channels, num_channels, kernel_size=3,
                                                stride=scale_factor, padding=1, output_padding=scale_factor - 1)

    def forward(self, x):
        if not self.use_deconv:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        else:
            return self.trans_conv(x)


class AttentionConnection(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.Tensor(1).fill_(factor))

    def forward(self, feature, attention):
        return (self.param + attention) * feature


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super().__init__()
        self.W_g = ConvNorm(F_g, F_int, kernel_size=1, stride=1, activation=False, **kwargs)

        self.W_x = ConvNorm(F_l, F_int, kernel_size=1, stride=2, activation=False, **kwargs)

        self.psi = nn.Sequential(
            ConvNorm(F_int, 1, kernel_size=1, stride=1, activation=False, **kwargs),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * self.upsample(psi)


class ParallelDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv3_0 = ConvNorm(in_channels[0], self.midchannels, 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], self.midchannels, 1, 1, **kwargs)
        self.conv5_0 = ConvNorm(in_channels[2], self.midchannels, 1, 1, **kwargs)

        self.conv4_5 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)
        self.conv3_4 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)

        self.conv_out = nn.Conv3d(3 * self.midchannels, out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        size = x3.shape[2:]

        # first interpolate three feature maps to the same resolution
        f3 = self.conv3_0(x3)  # (None, midchannels, h3, w3)
        f4 = self.conv4_0(F.interpolate(x4, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)
        level5 = self.conv5_0(F.interpolate(x5, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)

        # fuse feature maps
        level4 = self.conv4_5(torch.cat([f4, level5], dim=1))  # (None, midchannels, h3, w3)
        level3 = self.conv3_4(torch.cat([f3, level4], dim=1))  # (None, midchannels, h3, w3)

        fused_out_reduced = torch.cat([level3, level4, level5], dim=1)

        out = self.conv_out(fused_out_reduced)

        return out


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2

        self.conv5_4 = ConvNorm(in_channels[2], in_channels[1], 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], in_channels[1], 3, 1, **kwargs)
        self.conv4_3 = ConvNorm(in_channels[1], in_channels[0], 1, 1, **kwargs)
        self.conv3_0 = ConvNorm(in_channels[0], in_channels[0], 3, 1, **kwargs)

        self.conv_out = nn.Conv3d(in_channels[0], out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=False))
        x4_refine = self.conv4_0(x5_up + x4)
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='trilinear', align_corners=False))
        x3_refine = self.conv3_0(x4_up + x3)

        out = self.conv_out(x3_refine)

        return out






head_list = ['fcn', 'parallel']
head_map = {'fcn': FCNHead,
            'parallel': ParallelDecoder}


class Backbone(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.nb_filter = channels
        self.strides = strides + (5 - len(strides)) * (1,)
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        kwargs['norm'] = 'BATCH'

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = res_unit(input_channels, self.nb_filter[0], self.strides[0], **kwargs)
        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], **kwargs)
        self.conv4_0 = res_unit(self.nb_filter[3], self.nb_filter[4], self.strides[4], **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        raise NotImplementedError('Forward method must be implemented before calling it!')

    def predictor(self, x):
        return self.forward(x)['out']

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        if is_ds:
            loss_3 = criterion(outputs['level3'], targets)
            loss_2 = criterion(outputs['level2'], targets)
            loss_1 = criterion(outputs['level1'], targets)
            multi_loss = loss_out + loss_3 + loss_2 + loss_1
        else:
            multi_loss = loss_out
        return multi_loss


class UNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.up4_3(x4)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.up3_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.up2_1(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.up1_0(x1_3)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class AttentionUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

        self.gate4_3 = AttentionGate(nb_filter[4], nb_filter[3], nb_filter[3])
        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([self.gate4_3(x4, x3), self.up4_3(x4)], 1))
        x2_2 = self.conv2_2(torch.cat([self.gate3_2(x3_1, x2), self.up3_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([self.gate2_1(x2_2, x1), self.up2_1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([self.gate1_0(x1_3, x0), self.up1_0(x1_3)], 1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class CascadedUNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.first_stage = UNet(1, input_channels, channels, use_deconv, strides, **kwargs)
        self.second_stage = UNet(num_classes, input_channels, channels, use_deconv, strides, **kwargs)

    def forward(self, x):
        roi = self.first_stage(x)['out']
        roi_ = (torch.sigmoid(roi) > 0.5).float()
        roi_input = x * (1 + roi_)
        fine_seg = self.second_stage(roi_input)['out']
        output = dict()
        output['stage1'] = roi
        output['out'] = fine_seg
        return output

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        multi_loss = loss_out
        return multi_loss


class EnhancedUNet(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=3, channels=(32, 64, 128, 256, 512),
                 use_deconv=False, strides=(2, 2, 2, 2), **kwargs):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter
        strides = self.backbone.strides

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.up4_3(x4)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.up3_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.up2_1(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.up1_0(x1_3)], dim=1))

        out = dict()
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out


class zhao2022(SegmentationNetwork):
    '''3D Method

    Paper: https://ieeexplore.ieee.org/document/9852260
    
    Note: https://github.com/hsiangyuzhao/PANet/blob/main/models/networks.py

    # flops & params: 516.496G, 19.239M, 37.1529839038848 s
    '''
    def __init__(self, num_classes=4, head='fcn', input_channels=4, channels=(32, 64, 128, 256, 320),
                 use_deconv=False, strides=(1, 2, 2, 2, 2), leaky=True, net_arch=None, net_arch_dir=None, **kwargs):
        super(zhao2022, self).__init__()
        assert head in head_list
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        nb_filter = self.backbone.nb_filter

        self.head = head
        self.one_stage = head_map[head](in_channels=nb_filter[2:], out_channels=1)

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

        # downsample attention
        self.conv_down = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2)

        # upsample for the decoder
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1])
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2])
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3])
        self.up1_0 = ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-4], scale_factor=strides[-4])

        # parameterized skip connection
        self.skip_3 = AttentionConnection()
        self.skip_2 = AttentionConnection()
        self.skip_1 = AttentionConnection()
        self.skip_0 = AttentionConnection()

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        attention = self.one_stage(x2, x3, x4)

        act_attention = torch.sigmoid(attention)  # attention shape is the same as x2

        attention_x3 = self.conv_down(act_attention)
        attention3_1 = F.interpolate(attention_x3, size=x3.shape[2:], mode='trilinear', align_corners=False)
        attention2_2 = F.interpolate(act_attention, size=x2.shape[2:], mode='trilinear', align_corners=False)
        attention1_3 = F.interpolate(act_attention, size=x1.shape[2:], mode='trilinear', align_corners=False)
        attention0_4 = F.interpolate(act_attention, size=x0.shape[2:], mode='trilinear', align_corners=False)

        x3_1 = self.conv3_1(torch.cat([self.skip_3(x3, attention3_1), self.up4_3(x4)], dim=1))  # (nb_filter[3], H3, W3, D3)
        x2_2 = self.conv2_2(torch.cat([self.skip_2(x2, attention2_2), self.up3_2(x3_1)], dim=1))  # (nb_filter[2], H2, W2, D2)
        x1_3 = self.conv1_3(torch.cat([self.skip_1(x1, attention1_3), self.up2_1(x2_2)], dim=1))  # (nb_filter[1], H1, W1, D3)
        x0_4 = self.conv0_4(torch.cat([self.skip_0(x0, attention0_4), self.up1_0(x1_3)], dim=1))  # (nb_filter[0], H0, W0, D0)

        out = dict()
        out['stage1'] = F.interpolate(attention, size=size, mode='trilinear', align_corners=False)  # intermediate
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out