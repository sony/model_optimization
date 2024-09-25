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
    chunk, unbind, topk, gather, equal, transpose, permute, argmax, squeeze, multiply, subtract
from torch.nn import Conv2d, Linear, ConvTranspose2d, MaxPool2d
from torch.nn import Dropout, Flatten, Hardtanh
from torch.nn import ReLU, ReLU6, PReLU, SiLU, Sigmoid, Tanh, Hardswish, LeakyReLU
from torch.nn.functional import relu, relu6, prelu, silu, hardtanh, hardswish, leaky_relu

from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, PYTORCH_KERNEL, \
    BIAS
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tp_model import get_tp_model
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4 import __version__ as TPC_VERSION
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tp_model import OPSET_NO_QUANTIZATION, \
    OPSET_QUANTIZATION_PRESERVING, OPSET_DIMENSION_MANIPULATION_OPS_WITH_WEIGHTS, OPSET_DIMENSION_MANIPULATION_OPS, \
    OPSET_MERGE_OPS, OPSET_CONV, OPSET_FULLY_CONNECTED, OPSET_ANY_RELU, OPSET_ADD, OPSET_SUB, OPSET_MUL, OPSET_DIV, \
    OPSET_PRELU, OPSET_SWISH, OPSET_SIGMOID, OPSET_TANH

tp = mct.target_platform


def get_pytorch_tpc() -> tp.TargetPlatformCapabilities:
    """
    get a Pytorch TargetPlatformCapabilities object with default operation sets to layers mapping.

    Returns: a Pytorch TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    imx500_tpc_tp_model = get_tp_model()
    return generate_pytorch_tpc(name='imx500_tpc_pytorch_tpc', tp_model=imx500_tpc_tp_model)


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

    # we provide attributes mapping that maps each layer type in the operations set
    # that has weights attributes with provided quantization config (in the tp model) to
    # its framework-specific attribute name.
    # note that a DefaultDict should be provided if not all the layer types in the
    # operation set are provided separately in the mapping.
    pytorch_linear_attr_mapping = {KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                                   BIAS_ATTR: DefaultDict(default_value=BIAS)}

    with pytorch_tpc:
        tp.OperationsSetToLayers(OPSET_NO_QUANTIZATION, [torch.Tensor.size,
                                                         equal,
                                                         argmax,
                                                         topk])
        tp.OperationsSetToLayers(OPSET_QUANTIZATION_PRESERVING, [Dropout,
                                                                 dropout,
                                                                 split,
                                                                 chunk,
                                                                 unbind,
                                                                 MaxPool2d])
        tp.OperationsSetToLayers(OPSET_DIMENSION_MANIPULATION_OPS, [Flatten,
                                                                    flatten,
                                                                    operator.getitem,
                                                                    reshape,
                                                                    unsqueeze,
                                                                    squeeze,
                                                                    permute,
                                                                    transpose])
        tp.OperationsSetToLayers(OPSET_DIMENSION_MANIPULATION_OPS_WITH_WEIGHTS, [gather, torch.Tensor.expand])
        tp.OperationsSetToLayers(OPSET_MERGE_OPS,
                                 [torch.stack, torch.cat, torch.concat, torch.concatenate])

        tp.OperationsSetToLayers(OPSET_CONV, [Conv2d, ConvTranspose2d],
                                 attr_mapping=pytorch_linear_attr_mapping)
        tp.OperationsSetToLayers(OPSET_FULLY_CONNECTED, [Linear],
                                 attr_mapping=pytorch_linear_attr_mapping)
        tp.OperationsSetToLayers(OPSET_ANY_RELU, [torch.relu,
                                                  ReLU,
                                                  ReLU6,
                                                  LeakyReLU,
                                                  relu,
                                                  relu6,
                                                  leaky_relu,
                                                  tp.LayerFilterParams(Hardtanh, min_val=0),
                                                  tp.LayerFilterParams(hardtanh, min_val=0)])

        tp.OperationsSetToLayers(OPSET_ADD, [operator.add, add])
        tp.OperationsSetToLayers(OPSET_SUB, [operator.sub, sub, subtract])
        tp.OperationsSetToLayers(OPSET_MUL, [operator.mul, mul, multiply])
        tp.OperationsSetToLayers(OPSET_DIV, [operator.truediv, div, divide])
        tp.OperationsSetToLayers(OPSET_PRELU, [PReLU, prelu])
        tp.OperationsSetToLayers(OPSET_SWISH, [SiLU, silu, Hardswish, hardswish])
        tp.OperationsSetToLayers(OPSET_SIGMOID, [Sigmoid, sigmoid])
        tp.OperationsSetToLayers(OPSET_TANH, [Tanh, tanh])

    return pytorch_tpc
