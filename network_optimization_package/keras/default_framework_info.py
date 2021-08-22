# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
    MaxPooling2D, Activation, ReLU, GlobalAveragePooling2D, Add, Multiply, AveragePooling2D, UpSampling2D, InputLayer, \
    Concatenate, Softmax, PReLU

from network_optimization_package.common.quantization.quantization_config import QuantizationMethod
from network_optimization_package.common.defaultdict import DefaultDict
from network_optimization_package.common.framework_info import FrameworkInfo
from network_optimization_package.keras.constants import SOFTMAX, LINEAR, RELU, SWISH, SIGMOID, IDENTITY, TANH, SELU
from network_optimization_package.common.quantization.quantizers.symmetric_uniform_quantizer import symmetric_uniform_quantizer
from network_optimization_package.common.quantization.quantizers.kmeans_quantizer import kmeans_quantizer
from network_optimization_package.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from network_optimization_package.keras.quantizer.fake_quant_builder import constraint_quantization
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

