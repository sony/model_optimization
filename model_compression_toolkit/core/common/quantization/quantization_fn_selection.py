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

from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core.common.framework_info import get_fw_info
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_quantizer
from model_compression_toolkit.core.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer, \
    symmetric_quantizer, uniform_quantizer


def get_activation_quantization_fn_factory(quantization_method: QuantizationMethod) -> Callable[[int, dict], Callable]:
    """
    Get factory for activation quantizer.

    Args:
        quantization_method: quantization method for activation.

    Returns:
        Factory that accepts activation bitwidth and a dict of quantization params, and returns the quantizer.
    """
    return get_fw_info().activation_quantizer_factory_mapping[quantization_method]


def get_activation_quantization_fn(activation_quantization_cfg: NodeActivationQuantizationConfig) -> Callable:
    """
    Get activation quantizer based on activation quantization configuration.

    Args:
        activation_quantization_cfg: activation quantization configuration.

    Returns:
        Activation quantizer that accepts a tensor and returns a quantized tensor.
    """
    quantizer_factory = get_activation_quantization_fn_factory(
        activation_quantization_cfg.activation_quantization_method)
    quantizer = quantizer_factory(activation_quantization_cfg.activation_n_bits,
                                  activation_quantization_cfg.activation_quantization_params)
    return quantizer


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
