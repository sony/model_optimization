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


from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
    MaxPooling2D, Activation, ReLU, GlobalAveragePooling2D, Add, Multiply, AveragePooling2D, UpSampling2D, InputLayer, \
    Concatenate, Softmax, PReLU

from model_compression_toolkit.common.quantization.quantization_config import QuantizationMethod
from model_compression_toolkit.common.defaultdict import DefaultDict
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.keras.constants import SOFTMAX, LINEAR, RELU, SWISH, SIGMOID, IDENTITY, TANH, SELU
from model_compression_toolkit.common.quantization.quantizers.symmetric_uniform_quantizer import symmetric_uniform_quantizer
from model_compression_toolkit.common.quantization.quantizers.kmeans_quantizer import kmeans_quantizer
from model_compression_toolkit.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from model_compression_toolkit.keras.quantizer.fake_quant_builder import constraint_quantization
"""
Division of Keras layers by how they should be quantized.
KERNEL: Layers that their coefficients should be quantized.
ACTIVATION: Layers that their activation should be quantized.
NO_QUANTIZATION: Layers that should not be quantized.
"""

KERNEL = [Conv2D,
          DepthwiseConv2D,
          Dense,
          Conv2DTranspose]

NO_QUANTIZATION = [Reshape,
                   ZeroPadding2D,
                   Dropout,
                   MaxPooling2D] # TODO:  replace with marking

ACTIVATION = [Activation,
              ReLU,
              Softmax,
              GlobalAveragePooling2D,
              Add,
              Multiply,
              AveragePooling2D,
              UpSampling2D,
              InputLayer,
              Concatenate,
              PReLU]

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
                     SELU: (None, None)}

"""
Map from an Keras layer to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
LAYER2MINMAX = {Softmax: (0, 1),
                ReLU: (0, None)}

"""
Mapping from a QuantizationMethod to an activation quantizer function.
"""
ACTIVATION_QUANTIZER_MAPPING = {QuantizationMethod.SYMMETRIC_UNIFORM: constraint_quantization}

"""
Mapping from a QuantizationMethod to an weights quantizer function.
"""
WEIGHTS_QUANTIZER_MAPPING = {QuantizationMethod.SYMMETRIC_UNIFORM: symmetric_uniform_quantizer,
                             QuantizationMethod.KMEANS: kmeans_quantizer,
                             QuantizationMethod.LUT_QUANTIZER: lut_kmeans_quantizer}

DEFAULT_KERAS_INFO = FrameworkInfo(KERNEL,
                                   ACTIVATION,
                                   NO_QUANTIZATION,
                                   ACTIVATION_QUANTIZER_MAPPING,
                                   WEIGHTS_QUANTIZER_MAPPING,
                                   DEFAULT_CHANNEL_AXIS_DICT,
                                   ACTIVATION2MINMAX,
                                   LAYER2MINMAX)

