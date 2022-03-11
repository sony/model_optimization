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

from enum import Enum


class QuantizationMethod(Enum):
    """
    Method for quantization function selection:

    POWER_OF_TWO - Symmetric, uniform, threshold is power of two quantization.

    KMEANS - k-means quantization.

    LUT_QUANTIZER - quantization using a look up table.

    SYMMETRIC - Symmetric, uniform, quantization.

    UNIFORM - uniform quantization,

    """
    POWER_OF_TWO = 0
    KMEANS = 1
    LUT_QUANTIZER = 2
    SYMMETRIC = 3
    UNIFORM = 4


class HardwareModel:
    """
    Configure the hardware settings to use when optimizing the model.
    """

    def __init__(self,
                 activation_quantization_method: QuantizationMethod,
                 weights_quantization_method: QuantizationMethod):
        """

        Args:
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            weights_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for weights quantization.
        """

        self.activation_quantization_method = activation_quantization_method
        self.weights_quantization_method = weights_quantization_method

