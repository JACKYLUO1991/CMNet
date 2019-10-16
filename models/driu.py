# PyTorch implementation of DRIU:
# http://www.vision.ee.ethz.ch/~cvlsegmentation/driu/data/paper/DRIU_MICCAI2016.pdf

# MIT License

# Copyright (c) September 2018 Tim Laibacher

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models


class UpsampleBlock(nn.Module):
    def __init__(self, x_in_c, up_kernel_size, up_stride, up_padding):
        '''
        Args:
            img_h: input image height
            img_w: input image width 
            x_in_c : number of channels of intermediate layer
            up_kernel_size: kernel size for transposed convolution
            up_stride: stride for transposed convolution
            up_padding: padding for transposed convolution
        '''
        super().__init__()
        self.conv = nn.Conv2d(x_in_c, 16, 3, 1, 1)
        self.upconv = nn.ConvTranspose2d(
            16, 16, up_kernel_size, up_stride, up_padding)

    def forward(self, x_in, input_res):
        img_h = input_res[0]
        img_w = input_res[1]
        x = self.conv(x_in)
        x = self.upconv(x)
        # determine center crop
        # height
        up_h = x.shape[2]
        h_crop = up_h - img_h
        h_s = h_crop//2
        h_e = up_h - (h_crop - h_s)
        # width
        up_w = x.shape[3]
        w_crop = up_w-img_w
        w_s = w_crop//2
        w_e = up_w - (w_crop - w_s)

        # perform crop
        # needs explicit ranges for onnx export
        x = x[0:1, 0:16, h_s:h_e, w_s:w_e]  # crop to input size
        return x


class ConcatFuseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4*16, 1, 1, 1, 0)

    def forward(self, x1, x2, x3, x4):
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv(x_cat)
        return x


class DRIU(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG
        encoder = list(models.vgg16().children())[0][:23]
        self.conv1 = encoder[0:4]
        self.conv2 = encoder[4:9]
        self.conv3 = encoder[9:16]
        self.conv4 = encoder[16:23]

        # Upsample
        self.conv1_2_16 = nn.Conv2d(64, 16, 3, 1, 1)
        self.upsample2 = UpsampleBlock(128, 4, 2, 0)
        self.upsample4 = UpsampleBlock(256, 8, 4, 0)
        self.upsample8 = UpsampleBlock(512, 16, 8, 0)

        # Concat and Fuse
        self.concatfuse = ConcatFuseBlock()

    def forward(self, x):
        hw = x.shape[2:4]
        conv1 = self.conv1(x)  # conv1_2
        conv2 = self.conv2(conv1)  # conv2_2
        conv3 = self.conv3(conv2)  # conv3_3
        conv4 = self.conv4(conv3)  # conv4_3

        conv1_2_16 = self.conv1_2_16(conv1)    # conv1_2_16
        upsample2 = self.upsample2(conv2, hw)  # side-multi2-up
        upsample4 = self.upsample4(conv3, hw)  # side-multi3-up
        upsample8 = self.upsample8(conv4, hw)  # side-multi4-up
        out = self.concatfuse(conv1_2_16, upsample2, upsample4, upsample8)
        return F.sigmoid(out)
