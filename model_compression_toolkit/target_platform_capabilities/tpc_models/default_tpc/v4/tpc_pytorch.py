# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import add, sub, mul, div, flatten, reshape, split, unsqueeze, dropout, sigmoid, tanh, chunk, unbind, \
    permute, transpose, equal, gather, topk
from torch.nn import Conv2d, Linear, BatchNorm2d, ConvTranspose2d
from torch.nn import Dropout, Flatten, Hardtanh
from torch.nn import ReLU, ReLU6, PReLU, SiLU, Sigmoid, Tanh, Hardswish, LeakyReLU
from torch.nn.functional import relu, relu6, prelu, silu, hardtanh, hardswish, leaky_relu

from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.v3.tp_model import get_tp_model
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.v4 import __version__ as TPC_VERSION

tp = mct.target_platform


def get_pytorch_tpc() -> tp.TargetPlatformCapabilities:
    """
    get a Pytorch TargetPlatformCapabilities object with default operation sets to layers mapping.
    Returns: a Pytorch TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    default_tp_model = get_tp_model()
    return generate_pytorch_tpc(name='default_pytorch_tpc', tp_model=default_tp_model)


def generate_pytorch_tpc(name: str, tp_model: tp.TargetPlatformModel):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.
    Args:
        name: Name of the TargetPlatformModel.
        tp_model: TargetPlatformModel object.
    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    pytorch_tpc = tp.TargetPlatformCapabilities(tp_model,
                                                name=name,
                                                version=TPC_VERSION)

    with pytorch_tpc:
        tp.OperationsSetToLayers("NoQuantization", [Dropout,
                                                    Flatten,
                                                    dropout,
                                                    flatten,
                                                    split,
                                                    operator.getitem,
                                                    reshape,
                                                    unsqueeze,
                                                    BatchNorm2d,
                                                    chunk,
                                                    unbind,
                                                    torch.Tensor.size,
                                                    permute,
                                                    transpose,
                                                    equal,
                                                    gather,
                                                    topk])

        tp.OperationsSetToLayers("Conv", [Conv2d, ConvTranspose2d])
        tp.OperationsSetToLayers("FullyConnected", [Linear])
        tp.OperationsSetToLayers("AnyReLU", [torch.relu,
                                             ReLU,
                                             ReLU6,
                                             LeakyReLU,
                                             relu,
                                             relu6,
                                             leaky_relu,
                                             tp.LayerFilterParams(Hardtanh, min_val=0),
                                             tp.LayerFilterParams(hardtanh, min_val=0)])

        tp.OperationsSetToLayers("Add", [operator.add, add])
        tp.OperationsSetToLayers("Sub", [operator.sub, sub])
        tp.OperationsSetToLayers("Mul", [operator.mul, mul])
        tp.OperationsSetToLayers("Div", [operator.truediv, div])
        tp.OperationsSetToLayers("PReLU", [PReLU, prelu])
        tp.OperationsSetToLayers("Swish", [SiLU, silu, Hardswish, hardswish])
        tp.OperationsSetToLayers("Sigmoid", [Sigmoid, sigmoid])
        tp.OperationsSetToLayers("Tanh", [Tanh, tanh])

    return pytorch_tpc
