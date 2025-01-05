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
from model_compression_toolkit.target_platform_capabilities.target_platform import LayerFilterParams, Eq
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2fw import \
    AttachTpcToFramework


class AttachTpcToPytorch(AttachTpcToFramework):
    def __init__(self):
        super().__init__()

        self._opset2layer = {
            OperatorSetNames.OPSET_CONV: [Conv2d],
            OperatorSetNames.OPSET_DEPTHWISE_CONV: [],  # no specific operator for depthwise conv in pytorch
            OperatorSetNames.OPSET_CONV_TRANSPOSE: [ConvTranspose2d],
            OperatorSetNames.OPSET_FULLY_CONNECTED: [Linear],
            OperatorSetNames.OPSET_CONCATENATE: [torch.cat, torch.concat, torch.concatenate],
            OperatorSetNames.OPSET_STACK: [torch.stack],
            OperatorSetNames.OPSET_UNSTACK: [unbind],
            OperatorSetNames.OPSET_GATHER: [gather],
            OperatorSetNames.OPSET_EXPAND: [torch.Tensor.expand],
            OperatorSetNames.OPSET_BATCH_NORM: [BatchNorm2d],
            OperatorSetNames.OPSET_RELU: [torch.relu, ReLU, relu],
            OperatorSetNames.OPSET_RELU6: [ReLU6, relu6],
            OperatorSetNames.OPSET_LEAKY_RELU: [LeakyReLU, leaky_relu],
            OperatorSetNames.OPSET_HARD_TANH: [LayerFilterParams(Hardtanh, min_val=0),
                                                     LayerFilterParams(hardtanh, min_val=0)],
            OperatorSetNames.OPSET_ADD: [operator.add, add],
            OperatorSetNames.OPSET_SUB: [operator.sub, sub, subtract],
            OperatorSetNames.OPSET_MUL: [operator.mul, mul, multiply],
            OperatorSetNames.OPSET_DIV: [operator.truediv, div, divide],
            OperatorSetNames.OPSET_ADD_BIAS: [],  # no specific operator for bias_add in pytorch
            OperatorSetNames.OPSET_MIN: [minimum],
            OperatorSetNames.OPSET_MAX: [maximum],
            OperatorSetNames.OPSET_PRELU: [PReLU, prelu],
            OperatorSetNames.OPSET_SWISH: [SiLU, silu],
            OperatorSetNames.OPSET_SIGMOID: [Sigmoid, sigmoid, F.sigmoid],
            OperatorSetNames.OPSET_TANH: [Tanh, tanh, F.tanh],
            OperatorSetNames.OPSET_GELU: [GELU, gelu],
            OperatorSetNames.OPSET_HARDSIGMOID: [Hardsigmoid, hardsigmoid],
            OperatorSetNames.OPSET_HARDSWISH: [Hardswish, hardswish],
            OperatorSetNames.OPSET_FLATTEN: [Flatten, flatten],
            OperatorSetNames.OPSET_GET_ITEM: [operator.getitem],
            OperatorSetNames.OPSET_RESHAPE: [reshape],
            OperatorSetNames.OPSET_UNSQUEEZE: [unsqueeze],
            OperatorSetNames.OPSET_SQUEEZE: [squeeze],
            OperatorSetNames.OPSET_PERMUTE: [permute],
            OperatorSetNames.OPSET_TRANSPOSE: [transpose],
            OperatorSetNames.OPSET_DROPOUT: [Dropout, dropout],
            OperatorSetNames.OPSET_SPLIT_CHUNK: [split, chunk],
            OperatorSetNames.OPSET_MAXPOOL: [MaxPool2d, F.max_pool2d],
            OperatorSetNames.OPSET_AVGPOOL: [AvgPool2d, F.avg_pool2d],
            OperatorSetNames.OPSET_SIZE: [torch.Tensor.size],
            OperatorSetNames.OPSET_RESIZE: [torch.Tensor.resize],
            OperatorSetNames.OPSET_PAD: [F.pad],
            OperatorSetNames.OPSET_FOLD: [fold],
            OperatorSetNames.OPSET_SHAPE: [torch.Tensor.shape],
            OperatorSetNames.OPSET_EQUAL: [equal],
            OperatorSetNames.OPSET_ARGMAX: [argmax],
            OperatorSetNames.OPSET_TOPK: [topk],
            OperatorSetNames.OPSET_FAKE_QUANT: [fake_quantize_per_channel_affine],
            OperatorSetNames.OPSET_ZERO_PADDING2d: [ZeroPad2d],
            OperatorSetNames.OPSET_CAST: [torch.Tensor.type],
            OperatorSetNames.OPSET_STRIDED_SLICE: [],  # no such operator in pytorch, the equivalent is get_item which has a separate operator set
            OperatorSetNames.OPSET_ELU: [ELU, F.elu],
            OperatorSetNames.OPSET_SOFTMAX: [Softmax, softmax, F.softmax],
            OperatorSetNames.OPSET_LOG_SOFTMAX: [LogSoftmax],
            OperatorSetNames.OPSET_L2NORM: [LayerFilterParams(torch.nn.functional.normalize,
                                                                    Eq('p', 2) | Eq('p', None))],
            OperatorSetNames.OPSET_SSD_POST_PROCESS: [],  # no such operator in pytorch
            OperatorSetNames.OPSET_COMBINED_NON_MAX_SUPPRESSION: []  # no such operator in pytorch
        }

        pytorch_linear_attr_mapping = {KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                                       BIAS_ATTR: DefaultDict(default_value=BIAS)}
        self._opset2attr_mapping = {OperatorSetNames.OPSET_CONV: pytorch_linear_attr_mapping,
                                    OperatorSetNames.OPSET_CONV_TRANSPOSE: pytorch_linear_attr_mapping,
                                    OperatorSetNames.OPSET_FULLY_CONNECTED: pytorch_linear_attr_mapping}
