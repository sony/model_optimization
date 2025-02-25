# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import operator

import torch
from torch import add, sub, mul, div, divide, flatten, reshape, split, unsqueeze, dropout, sigmoid, tanh, \
    chunk, unbind, topk, gather, equal, transpose, permute, argmax, squeeze, multiply, subtract, minimum, \
    maximum, softmax, fake_quantize_per_channel_affine
from torch.nn import Conv2d, Linear, ConvTranspose2d, MaxPool2d, BatchNorm2d, Dropout, Flatten, Hardtanh, ReLU, ReLU6, \
    PReLU, SiLU, Sigmoid, Tanh, Hardswish, Hardsigmoid, LeakyReLU, GELU, LogSoftmax, Softmax, ELU, AvgPool2d, ZeroPad2d
from torch.nn.functional import relu, relu6, prelu, silu, hardtanh, hardswish, hardsigmoid, leaky_relu, gelu, fold
import torch.nn.functional as F

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, PYTORCH_KERNEL, BIAS, \
    BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2fw import \
    AttachTpcToFramework
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attribute_filter import Eq


class AttachTpcToPytorch(AttachTpcToFramework):
    def __init__(self):
        super().__init__()

        self._opset2layer = {
            OperatorSetNames.CONV: [Conv2d],
            OperatorSetNames.DEPTHWISE_CONV: [],  # no specific operator for depthwise conv in pytorch
            OperatorSetNames.CONV_TRANSPOSE: [ConvTranspose2d],
            OperatorSetNames.FULLY_CONNECTED: [Linear],
            OperatorSetNames.CONCATENATE: [torch.cat, torch.concat, torch.concatenate],
            OperatorSetNames.STACK: [torch.stack],
            OperatorSetNames.UNSTACK: [unbind],
            OperatorSetNames.GATHER: [gather],
            OperatorSetNames.EXPAND: [torch.Tensor.expand],
            OperatorSetNames.BATCH_NORM: [BatchNorm2d],
            OperatorSetNames.RELU: [torch.relu, ReLU, relu],
            OperatorSetNames.RELU6: [ReLU6, relu6],
            OperatorSetNames.LEAKY_RELU: [LeakyReLU, leaky_relu],
            OperatorSetNames.HARD_TANH: [LayerFilterParams(Hardtanh, min_val=0),
                                         LayerFilterParams(hardtanh, min_val=0)],
            OperatorSetNames.ADD: [operator.add, add],
            OperatorSetNames.SUB: [operator.sub, sub, subtract],
            OperatorSetNames.MUL: [operator.mul, mul, multiply],
            OperatorSetNames.DIV: [operator.truediv, div, divide],
            OperatorSetNames.ADD_BIAS: [],  # no specific operator for bias_add in pytorch
            OperatorSetNames.MIN: [minimum],
            OperatorSetNames.MAX: [maximum],
            OperatorSetNames.PRELU: [PReLU, prelu],
            OperatorSetNames.SWISH: [SiLU, silu],
            OperatorSetNames.SIGMOID: [Sigmoid, sigmoid, F.sigmoid],
            OperatorSetNames.TANH: [Tanh, tanh, F.tanh],
            OperatorSetNames.GELU: [GELU, gelu],
            OperatorSetNames.HARDSIGMOID: [Hardsigmoid, hardsigmoid],
            OperatorSetNames.HARDSWISH: [Hardswish, hardswish],
            OperatorSetNames.FLATTEN: [Flatten, flatten],
            OperatorSetNames.GET_ITEM: [operator.getitem],
            OperatorSetNames.RESHAPE: [reshape, torch.Tensor.view],
            OperatorSetNames.UNSQUEEZE: [unsqueeze],
            OperatorSetNames.SQUEEZE: [squeeze],
            OperatorSetNames.PERMUTE: [permute],
            OperatorSetNames.TRANSPOSE: [transpose],
            OperatorSetNames.DROPOUT: [Dropout, dropout],
            OperatorSetNames.SPLIT_CHUNK: [split, chunk],
            OperatorSetNames.MAXPOOL: [MaxPool2d, F.max_pool2d],
            OperatorSetNames.AVGPOOL: [AvgPool2d, F.avg_pool2d],
            OperatorSetNames.SIZE: [torch.Tensor.size],
            OperatorSetNames.RESIZE: [torch.Tensor.resize],
            OperatorSetNames.PAD: [F.pad],
            OperatorSetNames.FOLD: [fold],
            OperatorSetNames.SHAPE: [torch.Tensor.shape],
            OperatorSetNames.EQUAL: [equal],
            OperatorSetNames.ARGMAX: [argmax],
            OperatorSetNames.TOPK: [topk],
            OperatorSetNames.FAKE_QUANT: [fake_quantize_per_channel_affine],
            OperatorSetNames.ZERO_PADDING2D: [ZeroPad2d],
            OperatorSetNames.CAST: [torch.Tensor.type],
            OperatorSetNames.STRIDED_SLICE: [],  # no such operator in pytorch, the equivalent is get_item which has a separate operator set
            OperatorSetNames.ELU: [ELU, F.elu],
            OperatorSetNames.SOFTMAX: [Softmax, softmax, F.softmax],
            OperatorSetNames.LOG_SOFTMAX: [LogSoftmax],
            OperatorSetNames.L2NORM: [LayerFilterParams(torch.nn.functional.normalize,
                                                        Eq('p', 2) | Eq('p', None))],
            OperatorSetNames.SSD_POST_PROCESS: [],  # no such operator in pytorch
            OperatorSetNames.COMBINED_NON_MAX_SUPPRESSION: []  # no such operator in pytorch
        }

        pytorch_linear_attr_mapping = {KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                                       BIAS_ATTR: DefaultDict(default_value=BIAS)}
        self._opset2attr_mapping = {OperatorSetNames.CONV: pytorch_linear_attr_mapping,
                                    OperatorSetNames.CONV_TRANSPOSE: pytorch_linear_attr_mapping,
                                    OperatorSetNames.FULLY_CONNECTED: pytorch_linear_attr_mapping}
