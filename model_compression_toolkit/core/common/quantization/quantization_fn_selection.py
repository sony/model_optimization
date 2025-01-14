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

from collections.abc import Callable
from functools import partial

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from model_compression_toolkit.core.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer, \
    symmetric_quantizer, uniform_quantizer


def get_weights_quantization_fn(weights_quantization_method: QuantizationMethod) -> Callable:
    """
    Generate a function for weight quantization.

    Args:
        weights_quantization_method: Which quantization method to use for weights.
    Returns:
        A function to quantize a weight tensor.

    """

    if weights_quantization_method == QuantizationMethod.POWER_OF_TWO:
        quantizer_fn = power_of_two_quantizer
    elif weights_quantization_method == QuantizationMethod.SYMMETRIC:
        quantizer_fn = symmetric_quantizer
    elif weights_quantization_method == QuantizationMethod.UNIFORM:
        quantizer_fn = uniform_quantizer
    elif weights_quantization_method in [QuantizationMethod.LUT_POT_QUANTIZER, QuantizationMethod.LUT_SYM_QUANTIZER]:
        quantizer_fn = lut_kmeans_quantizer
    else:
        Logger.critical(
            f"No quantizer function found for the specified quantization method: {weights_quantization_method}")  # pragma: no cover

    return quantizer_fn
