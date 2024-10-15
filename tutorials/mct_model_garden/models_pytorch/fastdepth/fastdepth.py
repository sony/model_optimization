# ------------------------------------------------------------------------------
# This file contains code from the fast-depth repository.
#
# MIT License
#
# Copyright (c) 2019 Diana Wofk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# # ------------------------------------------------------------------------------

"""
Part of this code was based on fast-depth implementation. For more details, refer to the original repository:
https://github.com/dwofk/fast-depth
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

class MobileNetBackbone(nn.Module):
    def __init__(self, relu6=True):
        super(MobileNetBackbone, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(3, 16, 2, relu6),
            conv_dw(16, 56, 1, relu6),
            conv_dw(56, 88, 2, relu6),
            conv_dw(88, 120, 1, relu6),
            conv_dw(120, 144, 2, relu6),
            conv_dw(144, 256, 1, relu6),
            conv_dw(256, 408, 2, relu6),
            conv_dw(408, 376, 1, relu6),
            conv_dw(376, 272, 1, relu6),
            conv_dw(272, 288, 1, relu6),
            conv_dw(288, 296, 1, relu6),
            conv_dw(296, 328, 1, relu6),
            conv_dw(328, 480, 2, relu6),
            conv_dw(480, 512, 1, relu6),
            nn.AvgPool2d(7),
        )

class FastDepth(nn.Module, PyTorchModelHubMixin):
    def __init__(self):

        super(FastDepth, self).__init__()
        mobilenet = MobileNetBackbone()

        for i in range(14):
            setattr(self, 'conv{}'.format(i), mobilenet.model[i])

        kernel_size = 5
        self.decode_conv1 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 200))
        self.decode_conv2 = nn.Sequential(
            depthwise(200, kernel_size),
            pointwise(200, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 120))
        self.decode_conv4 = nn.Sequential(
            depthwise(120, kernel_size),
            pointwise(120, 56))
        self.decode_conv5 = nn.Sequential(
            depthwise(56, kernel_size),
            pointwise(56, 16))
        self.decode_conv6 = pointwise(16, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i == 1:
                x1 = x
            elif i == 3:
                x2 = x
            elif i == 5:
                x3 = x
        for i in range(1, 6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i == 4:
                x = x + x1
            elif i == 3:
                x = x + x2
            elif i == 2:
                x = x + x3
            # print("{}: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x

    def make_tensors_contiguous(self):
        for name, param in self.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        for name, buffer in self.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()

    def save_pretrained(self, save_directory, **kwargs):
        # Make tensors contiguous
        self.make_tensors_contiguous()
        # Call the original save_pretrained method
        super().save_pretrained(save_directory, **kwargs)



