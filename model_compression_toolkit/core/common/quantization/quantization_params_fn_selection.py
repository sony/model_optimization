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

from collections.abc import Callable
from functools import partial

from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.quantization.quantization_params_generation.kmeans_params import kmeans_tensor
from model_compression_toolkit.core.common.quantization.quantization_params_generation.lut_kmeans_params import \
    lut_kmeans_tensor, lut_kmeans_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import \
    symmetric_selection_tensor, symmetric_selection_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation.uniform_selection import \
    uniform_selection_histogram, uniform_selection_tensor
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import \
    power_of_two_selection_tensor, power_of_two_selection_histogram


def get_activation_quantization_params_fn(activation_quantization_method: QuantizationMethod) -> Callable:
    """
    Generate a function for finding activation quantization parameters.

    Args:
        activation_quantization_method: Which quantization method to use for activations.
    Returns:
        A function to find the quantization parameters.

    """
    if activation_quantization_method == QuantizationMethod.POWER_OF_TWO:
        params_fn = power_of_two_selection_histogram
    elif activation_quantization_method == QuantizationMethod.SYMMETRIC:
        params_fn = symmetric_selection_histogram
    elif activation_quantization_method == QuantizationMethod.UNIFORM:
        params_fn = uniform_selection_histogram
    elif activation_quantization_method == QuantizationMethod.LUT_POT_QUANTIZER:
        params_fn = lut_kmeans_histogram
    else:
        Logger.error(
            f'No params function for the configuration of '
            f'quantization method {activation_quantization_method}')  # pragma: no cover
    return params_fn


def get_weights_quantization_params_fn(weights_quantization_method: QuantizationMethod) -> Callable:
    """
    Generate a function for finding weights quantization parameters.

    Args:
        weights_quantization_method: Which quantization method to use for weights.
    Returns:
        A function to find the quantization parameters.

    """
    if weights_quantization_method == QuantizationMethod.POWER_OF_TWO:
        params_fn = power_of_two_selection_tensor
    elif weights_quantization_method == QuantizationMethod.SYMMETRIC:
        params_fn = symmetric_selection_tensor
    elif weights_quantization_method == QuantizationMethod.UNIFORM:
        params_fn = uniform_selection_tensor
    elif weights_quantization_method == QuantizationMethod.KMEANS:
        params_fn = kmeans_tensor
    elif weights_quantization_method == QuantizationMethod.LUT_POT_QUANTIZER:
        params_fn = partial(lut_kmeans_tensor, is_symmetric=False)
    elif weights_quantization_method == QuantizationMethod.LUT_SYM_QUANTIZER:
        params_fn = partial(lut_kmeans_tensor, is_symmetric=True)
    else:
        Logger.error(
            f'No params function for the configuration of '
            f'quantization method {weights_quantization_method}')  # pragma: no cover
    return params_fn
