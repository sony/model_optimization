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
from typing import Any
from functools import wraps

from torch.nn import Hardsigmoid, ReLU, ReLU6, Softmax, Sigmoid, GELU, SELU, SiLU
from torch.nn.functional import hardsigmoid, relu, relu6, softmax, gelu, selu, silu
from torch.nn import Conv2d, ConvTranspose2d, Linear
from torch import sigmoid

from model_compression_toolkit.core.common.framework_info import FrameworkInfo, set_fw_info, ChannelAxisMapping
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import SOFTMAX_THRESHOLD
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.quantizer.fake_quant_builder import power_of_two_quantization, \
    symmetric_quantization, uniform_quantization
from model_compression_toolkit.core.pytorch.quantizer.lut_fake_quant import activation_lut_kmean_quantizer


class PyTorchInfo(FrameworkInfo):
    """
    Extra field defined to handle Activation layer functions:
    """

    """
    Map each layer to it's weight attribute that should get quantized.
    If a layer that is not listed here is queried, None is returned.
    """
    kernel_ops_attribute_mapping = {Conv2d: KERNEL,
                                    ConvTranspose2d: KERNEL,
                                    Linear: KERNEL}

    """
    Map a layer to its kernel's output and input channels indices.
    Map's values are tuples of (output_channel_index, input_channel_index).
    Default value is returned for layers that are not included.
    """
    kernel_channels_mapping = {Conv2d: ChannelAxisMapping(0, 1),
                               Linear: ChannelAxisMapping(0, 1),
                               ConvTranspose2d: ChannelAxisMapping(1, 0)}

    """
    Map a layer to its output channel axis.
    Where axis=-1 is the last axis
    """
    out_channel_axis_mapping = {Conv2d: 1,
                                Linear: -1,
                                ConvTranspose2d: 1}

    """
    Map from an Pytorch module to its min/max output values (if known).
    The values are used for tensor min/max values initialization.
    """
    _layer_min_max_mapping = {Softmax: (0, SOFTMAX_THRESHOLD),
                              softmax: (0, SOFTMAX_THRESHOLD),
                              Sigmoid: (0, 1),
                              sigmoid: (0, 1),
                              Hardsigmoid: (0, 1),
                              hardsigmoid: (0, 1),
                              ReLU: (0, None),
                              relu: (0, None),
                              ReLU6: (0, None),
                              relu6: (0, None),
                              GELU: (-0.17, None),
                              gelu: (-0.17, None),
                              SELU: (-1.76, None),
                              selu: (-1.76, None),
                              silu: (-0.279, None),
                              SiLU: (-0.279, None),
                              }

    """
    Mapping from a QuantizationMethod to an activation quantizer function.
    """
    activation_quantizer_mapping = {QuantizationMethod.POWER_OF_TWO: power_of_two_quantization,
                                    QuantizationMethod.SYMMETRIC: symmetric_quantization,
                                    QuantizationMethod.UNIFORM: uniform_quantization,
                                    QuantizationMethod.LUT_POT_QUANTIZER: activation_lut_kmean_quantizer}

    @classmethod
    def get_kernel_channels(cls, node_type: Any) -> ChannelAxisMapping:
        """
        Returns node's channels mapping from kernel_channels_mapping or framework specific default value.
        Args:
            node_type: A node type.

        Returns:
            Node's channels mapping.

        """
        return cls.kernel_channels_mapping.get(node_type, cls._default_channel_mapping)

    @classmethod
    def get_out_channel_axis(cls, node_type: Any):
        """
        Returns node's output channel mapping from out_channel_axis_mapping or framework specific default value.
        Args:
            node_type: A node type.

        Returns:
            Node's output channel axis.

        """
        return cls.out_channel_axis_mapping.get(node_type, 1)


def set_pytorch_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        set_fw_info(PyTorchInfo)
        return func(*args, **kwargs)
    return wrapper
