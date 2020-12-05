import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .deform_conv_v2 import DeformConv2d


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


def _SplitChannels(channels, num_groups):
    # Channel evenly separated, if there is a remainder added to the first channel
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


def Conv3x3Bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# def Conv1x1Bn(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1 / ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(
            channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = nn.ReLU(inplace=True)
        self.se_expand = nn.Conv2d(
            squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y


class GroupedConv2d(nn.Module):
    '''Groupped convolution'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_size)
        self.channel_axis = 1
        self.split_in_channels = _SplitChannels(in_channels, self.num_groups)
        self.split_out_channels = _SplitChannels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels,
                              dim=self.channel_axis)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=self.channel_axis)

        return x


class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class MixNetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=[3],
            expand_ksize=[1],
            project_ksize=[1],
            stride=1,
            expand_ratio=1,
            se_ratio=0.0
    ):
        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (
                stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            pw_expansion = nn.Sequential(
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU(inplace=True)
            )
            conv.append(pw_expansion)

        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )
        conv.append(dw)

        if se:
            squeeze_excite = SqueezeAndExcite(
                expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        pw_projection = nn.Sequential(
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        # Bottleneck with expansion layer
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                      1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder block: upsample and concatenate with features maps from the encoder part
    """

    def __init__(self, up_in_c, x_in_c, upsamplemode='bilinear', expand_ratio=0.15):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode=upsamplemode, align_corners=False)  # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(
            up_in_c + x_in_c, (x_in_c + up_in_c) // 2, stride=1, expand_ratio=expand_ratio)

    def forward(self, up_in, x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = self.ir1(cat_x)
        return x


class LastDecoderBlock(nn.Module):
    def __init__(self, x_in_c, x_out_c, upsamplemode='bilinear', expand_ratio=0.15):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode=upsamplemode, align_corners=False)  # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(
            x_in_c, x_out_c, stride=1, expand_ratio=expand_ratio)

    def forward(self, up_in, x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = self.ir1(cat_x)
        return x


class MDMNet(nn.Module):
    '''A mixture of multiple convolution, with a large receptive field.At the same time, precision and speed balance'''

    def __init__(self, input_size, n_classes=1):
        super(MDMNet, self).__init__()
        # Make sure it's divisible by 16
        assert input_size[0] % 16 == 0 and input_size[1] % 16 == 0

        # encoder 1
        self.conv1 = Conv3x3Bn(
            in_channels=3, out_channels=32, stride=1)
        self.conv2 = Conv3x3Bn(
            in_channels=32, out_channels=16, stride=2)

        # encoder 2
        self.deformConv1 = DeformConv2d(
            inc=16, outc=16, kernel_size=3, padding=1, stride=1, bias=None, modulation=True)
        block_layer1 = []
        block_1_1 = MixNetBlock(16, 24, [3], [1, 1], [1,
                                                      1], stride=1, expand_ratio=3)
        block_1_2 = MixNetBlock(24, 40, [3], [1, 1], [
            1, 1], stride=2, expand_ratio=3)
        block_layer1.extend(
            [block_1_1, block_1_2])
        self.block1 = nn.Sequential(*block_layer1)

        # encoder 3
        self.deformConv2 = DeformConv2d(
            inc=40, outc=40, kernel_size=3, padding=1, stride=1, bias=None, modulation=True)
        block_layer2 = []
        block_2_1 = MixNetBlock(40, 80, [3, 5], [1, 1], [
            1, 1], stride=1, expand_ratio=3, se_ratio=.25)
        block_2_2 = MixNetBlock(80, 80, [3, 5], [1, 1], [
            1, 1], stride=1, expand_ratio=3, se_ratio=.25)
        block_2_3 = MixNetBlock(80, 80, [3, 5], [1], [
            1, 1], stride=2, expand_ratio=3, se_ratio=.25)
        block_layer2.extend([block_2_1, block_2_2, block_2_3])
        self.block2 = nn.Sequential(*block_layer2)

        # encoder 4
        self.deformConv3 = DeformConv2d(
            inc=80, outc=80, kernel_size=3, padding=1, stride=1, bias=None, modulation=True)
        block_layer3 = []
        block_3_1 = MixNetBlock(80, 80, [3, 5, 7], [1, 1], [
            1, 1], stride=1, expand_ratio=3, se_ratio=.25)
        block_3_2 = MixNetBlock(80, 120, [3, 5, 7], [1, 1], [
            1, 1], stride=1, expand_ratio=3, se_ratio=.25)
        block_3_3 = MixNetBlock(120, 120, [3, 5, 7], [1, 1], [
            1, 1], stride=2, expand_ratio=3, se_ratio=.25)
        block_layer3.extend([block_3_1, block_3_2, block_3_3])
        self.block3 = nn.Sequential(*block_layer3)

        self.decode4 = DecoderBlock(
            120, 80, upsamplemode='bilinear', expand_ratio=1)
        self.decode3 = DecoderBlock(
            100, 40, upsamplemode='bilinear', expand_ratio=1)
        self.decode2 = DecoderBlock(
            70, 16, upsamplemode='bilinear', expand_ratio=1)
        self.decode1 = LastDecoderBlock(
            46, x_out_c=n_classes, upsamplemode='bilinear', expand_ratio=1)

        # the side output
        self.side_4 = nn.Conv2d(
            100, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_3 = nn.Conv2d(
            70, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_2 = nn.Conv2d(
            43, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # Learnable parameters
        self.fu1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fu2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fu3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fu4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # Initial
        self.fu1.data.fill_(0.25)
        self.fu2.data.fill_(0.25)
        self.fu3.data.fill_(0.25)
        self.fu4.data.fill_(0.25)

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def forward(self, x):
        _, _, img_h, img_w = x.size()
        stem = self.conv2(self.conv1(x))

        # Encoder
        deform_conv1 = self.deformConv1(stem)
        b1 = self.block1(deform_conv1)

        deform_conv2 = self.deformConv2(b1)
        b2 = self.block2(deform_conv2)

        deform_conv3 = self.deformConv3(b2)
        b3 = self.block3(deform_conv3)

        # Decoder
        d4 = self.decode4(b3, b2)
        d3 = self.decode3(d4, b1)
        d2 = self.decode2(d3, stem)  # (b, 43, 256, 256)
        d1 = self.decode1(d2, x)

        # Model robustness
        d4_ = F.interpolate(d4, size=(img_h, img_w),
                            mode='bilinear', align_corners=False)
        d3_ = F.interpolate(d3, size=(img_h, img_w),
                            mode='bilinear', align_corners=False)
        d2_ = F.interpolate(d2, size=(img_h, img_w),
                            mode='bilinear', align_corners=False)

        side_4 = self.side_4(d4_)
        side_3 = self.side_3(d3_)
        side_2 = self.side_2(d2_)

        d = self.fu1 * d1 + self.fu2 * side_2 + self.fu3 * side_3 + self.fu4 * side_4

        # train d branch and test only d1 branch
        return [torch.sigmoid(d), torch.sigmoid(d1), torch.sigmoid(side_2), torch.sigmoid(side_3), \
            torch.sigmoid(side_4)]


if __name__ == '__main__':
    from torchstat import stat
    input_size = (784, 1168)

    net = MDMNet(input_size=input_size)
    stat(net, (3, ) + input_size)
