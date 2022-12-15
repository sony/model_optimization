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

from model_compression_toolkit.core.keras.quantizer.lut_fake_quant import activation_lut_kmean_quantizer
from packaging import version

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Softmax, ELU
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Softmax, ELU

from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core.common.framework_info import FrameworkInfo, ChannelAxis
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.constants import SOFTMAX_THRESHOLD
from model_compression_toolkit.core.keras.constants import SOFTMAX, LINEAR, RELU, SWISH, SIGMOID, IDENTITY, TANH, SELU, \
    KERNEL, DEPTHWISE_KERNEL
from model_compression_toolkit.core.keras.quantizer.fake_quant_builder import power_of_two_quantization, symmetric_quantization, uniform_quantization

"""
Map each layer to a list of its' weights attributes that should get quantized.
If a layer that is not listed here is queried, [None] is returned.
"""
KERNEL_ATTRIBUTES = DefaultDict({Conv2D: [KERNEL],
                                 DepthwiseConv2D: [DEPTHWISE_KERNEL],
                                 Dense: [KERNEL],
                                 Conv2DTranspose: [KERNEL]}, lambda: [None])


"""
Map a layer to its kernel's output and input channels indices.
Map's values are tuples of (output_channel_index, input_channel_index).
Default value is returned for layers that are not included.
"""
DEFAULT_CHANNEL_AXIS_DICT = DefaultDict({Conv2D: (3, 2),
                                         DepthwiseConv2D: (2, 2),
                                         Dense: (1, 0),
                                         Conv2DTranspose: (2, 3)}, lambda: (None, None))


"""
Map a layer to its output channel axis. 
Where axis=-1 is the last axis
"""
DEFAULT_OUT_CHANNEL_AXIS_DICT = DefaultDict({Conv2D: -1,
                                             DepthwiseConv2D: -1,
                                             Dense: -1,
                                             Conv2DTranspose: -1},
                                            lambda: -1)


"""
Map from an activation function to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
ACTIVATION2MINMAX = {SOFTMAX: (0, SOFTMAX_THRESHOLD),
                     SIGMOID: (0, 1),
                     LINEAR: (None, None),
                     IDENTITY: (None, None),
                     TANH: (-1, 1),
                     SWISH: (-0.279, None),
                     RELU: (0, None),
                     SELU: (None, None),
                     }

"""
Map from an Keras layer to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
LAYER2MINMAX = {Softmax: (0, SOFTMAX_THRESHOLD),
                ELU: (-1, None),
                tf.nn.silu: (-0.279, None),
                tf.nn.swish: (-0.279, None),
                tf.nn.sigmoid: (0, 1),
                tf.nn.tanh: (-1, 1),
                tf.nn.relu: (0, None),
                tf.nn.relu6: (0, 6),
                tf.nn.gelu: (-0.17, None),
                tf.nn.elu: (-1, None),
                tf.nn.selu: (-1.76, None),
                tf.nn.softplus: (0, None),
                tf.nn.softmax: (0, SOFTMAX_THRESHOLD),
                }
"""
Mapping from a QuantizationMethod to an activation quantizer function.
"""
ACTIVATION_QUANTIZER_MAPPING = {QuantizationMethod.POWER_OF_TWO: power_of_two_quantization,
                                QuantizationMethod.SYMMETRIC: symmetric_quantization,
                                QuantizationMethod.UNIFORM: uniform_quantization,
                                QuantizationMethod.LUT_POT_QUANTIZER: activation_lut_kmean_quantizer}


DEFAULT_KERAS_INFO = FrameworkInfo(ACTIVATION_QUANTIZER_MAPPING,
                                   DEFAULT_CHANNEL_AXIS_DICT,
                                   ACTIVATION2MINMAX,
                                   LAYER2MINMAX,
                                   KERNEL_ATTRIBUTES,
                                   DEFAULT_OUT_CHANNEL_AXIS_DICT)
