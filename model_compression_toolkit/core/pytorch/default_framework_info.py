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
from torch.nn import Hardsigmoid, ReLU, ReLU6, Softmax, Sigmoid
from torch.nn.functional import hardsigmoid, relu, relu6, softmax
from torch.nn import Conv2d, ConvTranspose2d, Linear
from torch import sigmoid

from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.constants import SOFTMAX_THRESHOLD
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.quantizer.fake_quant_builder import power_of_two_quantization, \
    symmetric_quantization, uniform_quantization
from model_compression_toolkit.core.pytorch.quantizer.lut_fake_quant import activation_lut_kmean_quantizer

"""
Map each layer to a list of its' weights attributes that should get quantized.
If a layer that is not listed here is queried, [None] is returned.
"""
KERNEL_ATTRIBUTES = DefaultDict({Conv2d: [KERNEL],
                                 ConvTranspose2d: [KERNEL],
                                 Linear: [KERNEL]},
                                [None])

"""
Map a layer to its kernel's output and input channels indices.
Map's values are tuples of (output_channel_index, input_channel_index).
Default value is returned for layers that are not included.
"""
DEFAULT_CHANNEL_AXIS_DICT = DefaultDict({Conv2d: (0, 1),
                                         Linear: (0, 1),
                                         ConvTranspose2d: (1, 0)},
                                        (None, None))

"""
Map a layer to its output channel axis.
Where axis=-1 is the last axis
"""
DEFAULT_OUT_CHANNEL_AXIS_DICT = DefaultDict({Conv2d: 1,
                                             Linear: -1,
                                             ConvTranspose2d: 1},
                                            1)


"""
Map from an activation function to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
ACTIVATION2MINMAX = {}  # should be an empty dict in Pytorch

"""
Map from an Pytorch module to its min/max output values (if known).
The values are used for tensor min/max values initialization.
"""
LAYER2MINMAX = {Softmax: (0, SOFTMAX_THRESHOLD),
                softmax: (0, SOFTMAX_THRESHOLD),
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
                                QuantizationMethod.UNIFORM: uniform_quantization,
                                QuantizationMethod.LUT_POT_QUANTIZER: activation_lut_kmean_quantizer}


DEFAULT_PYTORCH_INFO = FrameworkInfo(ACTIVATION_QUANTIZER_MAPPING,
                                     DEFAULT_CHANNEL_AXIS_DICT,
                                     ACTIVATION2MINMAX,
                                     LAYER2MINMAX,
                                     KERNEL_ATTRIBUTES,
                                     DEFAULT_OUT_CHANNEL_AXIS_DICT)
