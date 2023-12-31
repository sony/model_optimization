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

from collections.abc import Callable

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.quantization.quantizers.kmeans_quantizer import kmeans_quantizer
from model_compression_toolkit.core.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from model_compression_toolkit.core.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer, \
    symmetric_quantizer, uniform_quantizer


def raise_exception(quantization_method: Any):
    """
    Function that is used when quantization method is not supported. Using this function,
    an error is raised when actually trying to use and call this function (and not when a
    quantization function is retrieved).
    """
    Logger.error(f'No quantizer function for the configuration of quantization method {quantization_method}')

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
    elif weights_quantization_method == QuantizationMethod.KMEANS:
        quantizer_fn = kmeans_quantizer
    elif weights_quantization_method in [QuantizationMethod.LUT_POT_QUANTIZER, QuantizationMethod.LUT_SYM_QUANTIZER]:
        quantizer_fn = lut_kmeans_quantizer
    else:
        quantizer_fn = lambda *args, **kwargs: raise_exception(weights_quantization_method)
    return quantizer_fn



def get_activations_quantization_fn(activations_quantization_method: QuantizationMethod,
                                    fw_info: FrameworkInfo) -> Callable:
    """
    Get function for activation quantization. It is taken from framework info if it's defined
    there, or a function that raises an exception is returned.

    Args:
        activations_quantization_method: Activation quantization method to retrieve its function.
        fw_info: Framework info to get the quantization function from.

    Returns:
        Function to use when quantizing activations.
    """

    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(activations_quantization_method)
    if activation_quantization_fn is None:
        activation_quantization_fn = lambda *args, **kwargs: raise_exception(activations_quantization_method)
    return activation_quantization_fn