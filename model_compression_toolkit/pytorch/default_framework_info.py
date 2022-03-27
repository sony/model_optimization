# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import torch
from torch.nn import Hardswish, Hardsigmoid, ReLU, Hardtanh, ReLU6, LeakyReLU, PReLU, SiLU, Softmax, \
    Sigmoid, Softplus, Softsign, Tanh
from torch.nn.functional import hardswish, hardsigmoid, relu, hardtanh, relu6, leaky_relu, prelu, silu, softmax, \
    softplus, softsign
from torch.nn import UpsamplingBilinear2d, AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from torch.nn.functional import upsample_bilinear, adaptive_avg_pool2d, avg_pool2d, max_pool2d
from torch.nn import Conv2d, ConvTranspose2d, Linear, BatchNorm2d
from torch.nn import Dropout, Flatten
from torch import add, multiply, mul, sub, flatten, reshape, split, unsqueeze, concat, cat,\
    mean, dropout, sigmoid, tanh
import operator

from model_compression_toolkit.common.defaultdict import DefaultDict
from model_compression_toolkit.common.framework_info import FrameworkInfo, ChannelAxis
from model_compression_toolkit.common.hardware_representation import QuantizationMethod
from model_compression_toolkit.common.quantization.quantizers.kmeans_quantizer import kmeans_quantizer
from model_compression_toolkit.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from model_compression_toolkit.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer, \
    symmetric_quantizer, uniform_quantizer
from model_compression_toolkit.pytorch.constants import KERNEL
from model_compression_toolkit.pytorch.quantizer.fake_quant_builder import power_of_two_quantization, \
    symmetric_quantization, uniform_quantization
from model_compression_toolkit.pytorch.quantizer.fake_quant_builder import power_of_two_quantization
from model_compression_toolkit.pytorch.reader.graph_builders import DummyPlaceHolder, ConstantHolder

"""
Division of Pytorch modules by how they should be quantized.
KERNEL_OPS: Layers that their coefficients should be quantized.
ACTIVATION: Layers that their activation should be quantized.
NO_QUANTIZATION: Layers that should not be quantized.
"""

KERNEL_OPS = [Conv2d, Linear, ConvTranspose2d]

NO_QUANTIZATION = [Dropout, Flatten, ConstantHolder] + \
                  [dropout, flatten, split, operator.getitem, reshape, unsqueeze]

ACTIVATION = [DummyPlaceHolder] + \
    [Hardswish, Hardsigmoid, ReLU, Hardtanh, ReLU6, LeakyReLU, PReLU, SiLU, Softmax, Sigmoid, Softplus, Softsign, Tanh] + \
    [hardswish, hardsigmoid, relu, hardtanh, relu6, leaky_relu, prelu, silu, softmax, sigmoid, softplus, softsign, tanh] + \
    [torch.relu] + \
    [UpsamplingBilinear2d, AdaptiveAvgPool2d, AvgPool2d, MaxPool2d] + \
    [upsample_bilinear, adaptive_avg_pool2d, avg_pool2d, max_pool2d] + \
    [add, sub, mul, multiply] + \
    [operator.add, operator.sub, operator.mul] + \
    [BatchNorm2d, concat, cat, mean]

"""
Map each layer to a list of its' weights attributes that should get quantized.
If a layer that is not listed here is queried, [None] is returned.
"""
KERNEL_ATTRIBUTES = DefaultDict({Conv2d: [KERNEL],
                                 ConvTranspose2d: [KERNEL],
                                 Linear: [KERNEL]},
                                lambda: [None])

"""
Map a layer to its kernel's output and input channels indices.
Map's values are tuples of (output_channel_index, input_channel_index).
Default value is returned for layers that are not included.
"""
DEFAULT_CHANNEL_AXIS_DICT = DefaultDict({Conv2d: (0, 1),
                                         Linear: (0, 1),
                                         ConvTranspose2d: (1, 0)},
                                        lambda: (None, None))

"""
Map from an activation function to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
ACTIVATION2MINMAX = {} # should be an empty dict in Pytorch

"""
Map from an Pytorch module to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
LAYER2MINMAX = {Softmax: (0, 1),
                softmax: (0, 1),
                Sigmoid: (0, 1),
                sigmoid: (0, 1),
                Hardsigmoid: (0, 1),
                hardsigmoid: (0, 1),
                ReLU: (0, None),
                relu: (0, None),
                ReLU6: (0, None),
                relu6: (0, None)}

"""
Mapping from a QuantizationMethod to an activation quantizer function.
"""
ACTIVATION_QUANTIZER_MAPPING = {QuantizationMethod.POWER_OF_TWO: power_of_two_quantization,
                                QuantizationMethod.SYMMETRIC: symmetric_quantization,
                                QuantizationMethod.UNIFORM: uniform_quantization}

"""
Mapping from a QuantizationMethod to an weights quantizer function.
"""
WEIGHTS_QUANTIZER_MAPPING = {QuantizationMethod.POWER_OF_TWO: power_of_two_quantizer,
                             QuantizationMethod.SYMMETRIC: symmetric_quantizer,
                             QuantizationMethod.UNIFORM: uniform_quantizer,
                             QuantizationMethod.KMEANS: kmeans_quantizer,
                             QuantizationMethod.LUT_QUANTIZER: lut_kmeans_quantizer}

"""
Output channel index of the model's layers
"""
OUTPUT_CHANNEL_INDEX = ChannelAxis.NCHW

DEFAULT_PYTORCH_INFO = FrameworkInfo(KERNEL_OPS,
                                     ACTIVATION,
                                     NO_QUANTIZATION,
                                     ACTIVATION_QUANTIZER_MAPPING,
                                     WEIGHTS_QUANTIZER_MAPPING,
                                     DEFAULT_CHANNEL_AXIS_DICT,
                                     ACTIVATION2MINMAX,
                                     LAYER2MINMAX,
                                     KERNEL_ATTRIBUTES,
                                     OUTPUT_CHANNEL_INDEX)
