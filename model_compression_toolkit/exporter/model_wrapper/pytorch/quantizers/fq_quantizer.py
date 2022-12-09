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
# ==============================================================================\
import numpy as np
from typing import Dict, Any

from model_compression_toolkit.core.common.constants import RANGE_MIN, RANGE_MAX
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.pytorch.quantizer.fake_quant_builder import uniform_quantization


class FakeQuantQuantizer:
    """
    Quantizer using TensorFlow fake quant layer to quantize activations.
    """

    def __init__(self,
                 nbits: int,
                 min_range: np.ndarray,
                 max_range: np.ndarray,
                 quantization_method: QuantizationMethod):
        """

        Args:
            nbits: Number of bits to quantize.
            min_range: Min quantization range.
            max_range: Max quantization range.
            quantization_method: Quantization method that is used (POT, Uniform, etc.)

        """
        self.nbits = nbits
        self.min_range = min_range
        self.max_range = max_range
        self.quantization_method = quantization_method

    def __call__(self, inputs):
        """
        Apply quantization to the input tensor.

        Args:
            inputs: Input tensor to be quantized.

        Returns:
            Quantized tensor.
        """
        return uniform_quantization(self.nbits,
                                    {RANGE_MIN: self.min_range,
                                     RANGE_MAX: self.max_range})(inputs)
