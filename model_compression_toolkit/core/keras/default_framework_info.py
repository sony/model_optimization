# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

import tensorflow as tf

from typing import Tuple, Any, Dict
from functools import wraps

from model_compression_toolkit.core.keras.quantizer.lut_fake_quant import activation_lut_kmean_quantizer
from packaging import version

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Softmax, ELU, Activation
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Softmax, ELU, Activation  # pragma: no cover
from model_compression_toolkit.core.common.framework_info import FrameworkInfo, set_fw_info, ChannelAxisMapping
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import SOFTMAX_THRESHOLD, ACTIVATION
from model_compression_toolkit.core.keras.constants import SOFTMAX, LINEAR, RELU, SWISH, SIGMOID, IDENTITY, TANH, SELU, \
    KERNEL, DEPTHWISE_KERNEL, GELU
from model_compression_toolkit.core.keras.quantizer.fake_quant_builder import power_of_two_quantization, symmetric_quantization, uniform_quantization


class KerasInfo(FrameworkInfo):
    """
    Extra field defined to handle Activation layer functions:

        _activation_min_max_mapping (Dict[str, tuple]): Dictionary from an activation function to its min/max output values.

    """

    """
    Map each layer to it's weight attribute that should get quantized.
    If a layer that is not listed here is queried, None is returned.
    """
    kernel_ops_attribute_mapping = {Conv2D: KERNEL,
                                    DepthwiseConv2D: DEPTHWISE_KERNEL,
                                    Dense: KERNEL,
                                    Conv2DTranspose: KERNEL}

    """
    Map a layer to its kernel's output and input channels indices.
    Map's values are tuples of (output_channel_index, input_channel_index).
    Default value is returned for layers that are not included.
    """
    kernel_channels_mapping = {Conv2D: ChannelAxisMapping(3, 2),
                               DepthwiseConv2D: ChannelAxisMapping(2, 2),
                               Dense: ChannelAxisMapping(1, 0),
                               Conv2DTranspose: ChannelAxisMapping(2, 3)}

    """
    Map a layer to its output channel axis.
    Where axis=-1 is the last axis
    """
    out_channel_axis_mapping = {Conv2D: -1,
                                DepthwiseConv2D: -1,
                                Dense: -1,
                                Conv2DTranspose: -1}

    """
    Map from an activation function name to its min/max output values (if known).
    The values are used for tensor min/max values initialization.
    """
    _activation_min_max_mapping = {SOFTMAX: (0, SOFTMAX_THRESHOLD),
                                   SIGMOID: (0, 1),
                                   LINEAR: (None, None),
                                   IDENTITY: (None, None),
                                   TANH: (-1, 1),
                                   SWISH: (-0.279, None),
                                   RELU: (0, None),
                                   SELU: (-1.76, None),
                                   GELU: (-0.17, None),
                                   }

    """
    Map from an Keras module to its min/max output values (if known).
    The values are used for tensor min/max values initialization.
    """
    _layer_min_max_mapping = {Softmax: (0, SOFTMAX_THRESHOLD),
                              ELU: (-1, None),
                              tf.nn.silu: (-0.279, None),
                              tf.nn.swish: (-0.279, None),
                              tf.nn.sigmoid: (0, 1),
                              tf.nn.tanh: (-1, 1),
                              tf.nn.relu: (0, None),
                              tf.nn.relu6: (0, None),
                              tf.nn.gelu: (-0.17, None),
                              tf.nn.elu: (-1, None),
                              tf.nn.selu: (-1.76, None),
                              tf.nn.softplus: (0, None),
                              tf.nn.softmax: (0, SOFTMAX_THRESHOLD),
                              }

    """
    Mapping from a QuantizationMethod to an activation quantizer function.
    """
    activation_quantizer_mapping = {QuantizationMethod.POWER_OF_TWO: power_of_two_quantization,
                                    QuantizationMethod.SYMMETRIC: symmetric_quantization,
                                    QuantizationMethod.UNIFORM: uniform_quantization,
                                    QuantizationMethod.LUT_POT_QUANTIZER: activation_lut_kmean_quantizer}

    @classmethod
    def get_layer_min_max(cls, layer: Any, fw_attrs: Dict) -> Tuple[float, float]:
        """
        Return layer min/max mapping the FrameworkInfo holds.
        Args:
            layer: A layer to check if has a min/max known values.
            fw_attrs: framework attributes from framework layer.

        Returns:
            Layer's min/max known values.
        """

        if cls.layers_has_min_max(layer):
            return cls._layer_min_max_mapping[layer]
        elif isinstance(layer, Activation) and fw_attrs[ACTIVATION] in cls._activation_min_max_mapping:
            return cls._activation_min_max_mapping[fw_attrs[ACTIVATION]]
        else:
            return None, None

    @classmethod
    def get_kernel_channels(cls, node_type: Any) -> ChannelAxisMapping:
        """
        Returns node's channels mapping from kernel_channels_mapping or framework specific default value.
        Args:
            node_type: A node type

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
        return cls.out_channel_axis_mapping.get(node_type, -1)


def set_keras_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        set_fw_info(KerasInfo)
        return func(*args, **kwargs)
    return wrapper
