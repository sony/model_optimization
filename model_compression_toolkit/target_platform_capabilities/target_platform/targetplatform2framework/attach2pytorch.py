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
    maximum
from torch.nn import Conv2d, Linear, ConvTranspose2d, MaxPool2d, BatchNorm2d
from torch.nn import Dropout, Flatten, Hardtanh
from torch.nn import ReLU, ReLU6, PReLU, SiLU, Sigmoid, Tanh, Hardswish, Hardsigmoid, LeakyReLU, GELU
import torch.nn.functional as F
from torch.nn.functional import relu, relu6, prelu, silu, hardtanh, hardswish, hardsigmoid, leaky_relu, gelu

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, PYTORCH_KERNEL, BIAS, \
    BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames
from model_compression_toolkit.target_platform_capabilities.target_platform import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2fw import \
    AttachTpModelToFw


class AttachTpModelToPytorch(AttachTpModelToFw):
    def __init__(self):
        super().__init__()

        self._opset2layer = {
            OperatorSetNames.OPSET_CONV.value: [Conv2d],
            OperatorSetNames.OPSET_CONV_TRANSPOSE.value: [ConvTranspose2d],
            OperatorSetNames.OPSET_FULLY_CONNECTED.value: [Linear],
            OperatorSetNames.OPSET_CONCATENATE.value: [torch.cat, torch.concat, torch.concatenate],
            OperatorSetNames.OPSET_STACK.value: [torch.stack],
            OperatorSetNames.OPSET_UNSTACK.value: [unbind],
            OperatorSetNames.OPSET_GATHER.value: [gather],
            OperatorSetNames.OPSET_EXPAND.value: [torch.Tensor.expand],
            OperatorSetNames.OPSET_BATCH_NORM.value: [BatchNorm2d],
            OperatorSetNames.OPSET_RELU.value: [torch.relu, ReLU, relu],
            OperatorSetNames.OPSET_RELU6.value: [ReLU6, relu6],
            OperatorSetNames.OPSET_LEAKY_RELU.value: [LeakyReLU, leaky_relu],
            OperatorSetNames.OPSET_HARD_TANH.value: [LayerFilterParams(Hardtanh, min_val=0),
                                                     LayerFilterParams(hardtanh, min_val=0)],
            OperatorSetNames.OPSET_ADD.value: [operator.add, add],
            OperatorSetNames.OPSET_SUB.value: [operator.sub, sub, subtract],
            OperatorSetNames.OPSET_MUL.value: [operator.mul, mul, multiply],
            OperatorSetNames.OPSET_DIV.value: [operator.truediv, div, divide],
            OperatorSetNames.OPSET_MIN.value: [minimum],
            OperatorSetNames.OPSET_MAX.value: [maximum],
            OperatorSetNames.OPSET_PRELU.value: [PReLU, prelu],
            OperatorSetNames.OPSET_SWISH.value: [SiLU, silu],
            OperatorSetNames.OPSET_SIGMOID.value: [Sigmoid, sigmoid, F.sigmoid],
            OperatorSetNames.OPSET_TANH.value: [Tanh, tanh, F.tanh],
            OperatorSetNames.OPSET_GELU.value: [GELU, gelu],
            OperatorSetNames.OPSET_HARDSIGMOID.value: [Hardsigmoid, hardsigmoid],
            OperatorSetNames.OPSET_HARDSWISH.value: [Hardswish, hardswish],
            OperatorSetNames.OPSET_FLATTEN.value: [Flatten, flatten],
            OperatorSetNames.OPSET_GET_ITEM.value: [operator.getitem],
            OperatorSetNames.OPSET_RESHAPE.value: [reshape],
            OperatorSetNames.OPSET_UNSQUEEZE.value: [unsqueeze],
            OperatorSetNames.OPSET_SQUEEZE.value: [squeeze],
            OperatorSetNames.OPSET_PERMUTE.value: [permute],
            OperatorSetNames.OPSET_TRANSPOSE.value: [transpose],
            OperatorSetNames.OPSET_DROPOUT.value: [Dropout, dropout],
            OperatorSetNames.OPSET_SPLIT.value: [split],
            OperatorSetNames.OPSET_CHUNK.value: [chunk],
            OperatorSetNames.OPSET_MAXPOOL.value: [MaxPool2d],
            OperatorSetNames.OPSET_SIZE.value: [torch.Tensor.size],
            OperatorSetNames.OPSET_SHAPE.value: [torch.Tensor.shape],
            OperatorSetNames.OPSET_EQUAL.value: [equal],
            OperatorSetNames.OPSET_ARGMAX.value: [argmax],
            OperatorSetNames.OPSET_TOPK.value: [topk],
        }

        pytorch_linear_attr_mapping = {KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                                       BIAS_ATTR: DefaultDict(default_value=BIAS)}
        self._opset2attr_mapping = {OperatorSetNames.OPSET_CONV.value: pytorch_linear_attr_mapping,
                                    OperatorSetNames.OPSET_CONV_TRANSPOSE.value: pytorch_linear_attr_mapping,
                                    OperatorSetNames.OPSET_FULLY_CONNECTED.value: pytorch_linear_attr_mapping}
