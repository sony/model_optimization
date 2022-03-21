# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, GlobalAveragePooling2D, Add, Multiply, AveragePooling2D, UpSampling2D, InputLayer, \
        Concatenate, Softmax, PReLU, Flatten, Cropping2D, ELU, Dot, LeakyReLU, Permute, LayerNormalization
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, GlobalAveragePooling2D, Add, Multiply, AveragePooling2D, UpSampling2D, \
    InputLayer, Concatenate, Softmax, PReLU, Flatten, Cropping2D, Dot, ELU, LeakyReLU, Permute, LayerNormalization

from model_compression_toolkit.common.defaultdict import DefaultDict
from model_compression_toolkit.common.framework_info import FrameworkInfo, ChannelAxis
from model_compression_toolkit.common.hardware_representation import QuantizationMethod
from model_compression_toolkit.common.quantization.quantizers.kmeans_quantizer import kmeans_quantizer
from model_compression_toolkit.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from model_compression_toolkit.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer, \
    symmetric_quantizer, uniform_quantizer
from model_compression_toolkit.keras.constants import SOFTMAX, LINEAR, RELU, SWISH, SIGMOID, IDENTITY, TANH, SELU, \
    KERNEL, DEPTHWISE_KERNEL
from model_compression_toolkit.keras.quantizer.fake_quant_builder import power_of_two_quantization, symmetric_quantization, uniform_quantization

"""
Division of Keras layers by how they should be quantized.
KERNEL_OPS: Layers that their coefficients should be quantized.
ACTIVATION: Layers that their activation should be quantized.
NO_QUANTIZATION: Layers that should not be quantized.
"""

KERNEL_OPS = [Conv2D,
              DepthwiseConv2D,
              Dense,
              Conv2DTranspose]

NO_QUANTIZATION = [Reshape,
                   tf.reshape,
                   Flatten,
                   Permute,
                   Cropping2D,
                   ZeroPadding2D,
                   Dropout,
                   MaxPooling2D,
                   tf.reshape,
                   tf.split,
                   tf.quantization.fake_quant_with_min_max_vars]  # TODO:  replace with marking

ACTIVATION = [Activation,
              ReLU,
              tf.nn.relu,
              tf.nn.relu6,
              tf.nn.leaky_relu,
              Softmax,
              GlobalAveragePooling2D,
              Add,
              Multiply,
              AveragePooling2D,
              UpSampling2D,
              InputLayer,
              Concatenate,
              PReLU,
              ELU,
              tf.nn.silu,
              tf.nn.swish,
              tf.nn.sigmoid,
              tf.nn.tanh,
              tf.nn.relu,
              tf.nn.relu6,
              tf.nn.leaky_relu,
              LeakyReLU,
              tf.nn.softsign,
              tf.nn.gelu,
              tf.nn.elu,
              tf.nn.selu,
              tf.nn.softplus,
              tf.nn.softmax,
              Dot,
              LayerNormalization,
              tf.add,
              tf.multiply,
              tf.reduce_mean,
              tf.reduce_min,
              tf.reduce_sum,
              tf.reduce_max,
              tf.image.resize,
              tf.image.crop_and_resize,
              tf.concat,
              ]




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
Map from an activation function to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
ACTIVATION2MINMAX = {SOFTMAX: (0, 1),
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
LAYER2MINMAX = {Softmax: (0, 1),
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
                tf.nn.softmax: (0, 1),
                }
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
OUTPUT_CHANNEL_INDEX = ChannelAxis.NHWC

DEFAULT_KERAS_INFO = FrameworkInfo(KERNEL_OPS,
                                   ACTIVATION,
                                   NO_QUANTIZATION,
                                   ACTIVATION_QUANTIZER_MAPPING,
                                   WEIGHTS_QUANTIZER_MAPPING,
                                   DEFAULT_CHANNEL_AXIS_DICT,
                                   ACTIVATION2MINMAX,
                                   LAYER2MINMAX,
                                   KERNEL_ATTRIBUTES,
                                   OUTPUT_CHANNEL_INDEX)
