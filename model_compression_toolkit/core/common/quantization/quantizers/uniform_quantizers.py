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

import numpy as np

from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.constants import RANGE_MIN, RANGE_MAX, THRESHOLD
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import uniform_quantize_tensor, \
    quantize_tensor


def threshold_is_power_of_two(threshold: np.ndarray, per_channel: bool) -> bool:
    if per_channel:
        thresholds_per_channel = threshold.flatten()
        return (np.log2(thresholds_per_channel) == list(map(int, np.log2(thresholds_per_channel)))).all()

    return np.log2(threshold) == int(np.log2(threshold))


def power_of_two_quantizer(tensor_data: np.ndarray,
                           n_bits: int,
                           signed: bool,
                           quantization_params: dict,
                           per_channel: bool,
                           output_channels_axis: int) -> np.ndarray:
    """
    Quantize a tensor according to given: threshold, number of bits, and whether
    quantization range is sign or unsigned.

    Args:
        tensor_data: Tensor values to quantize.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not.
        quantization_params: Dictionary of specific parameters for this quantization function.
        per_channel: Whether to use separate quantization per output channel.
        output_channels_axis: Axis of the output channel.

    Returns:
        Quantized data.
    """
    threshold = quantization_params.get(THRESHOLD)
    if threshold is None:
        Logger.error(f"{THRESHOLD} parameter must be defined in 'quantization_params'")  # pragma: no cover
    if not threshold_is_power_of_two(threshold, per_channel):
        Logger.error(f"Expects {THRESHOLD} parameter to be a power of two, but got {threshold}")  # pragma: no cover

    return quantize_tensor(tensor_data,
                           threshold,
                           n_bits,
                           signed)


def symmetric_quantizer(tensor_data: np.ndarray,
                        n_bits: int,
                        signed: bool,
                        quantization_params: dict,
                        per_channel: bool,
                        output_channels_axis: int) -> np.ndarray:
    """
    Quantize a tensor in a symmetric range according to given: threshold, number of bits, and whether
    quantization range is symmetric and is sign or unsigned.

    Args:
        tensor_data: Tensor values to quantize.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not.
        quantization_params: Dictionary of specific parameters for this quantization function.
        per_channel: Whether to use separate quantization per output channel.
        output_channels_axis: Axis of the output channel.

    Returns:
        Quantized data.
    """
    threshold = quantization_params.get(THRESHOLD)
    if threshold is None:
        Logger.error(f"{THRESHOLD} parameter must be defined in 'quantization_params'")  # pragma: no cover

    return quantize_tensor(tensor_data,
                           threshold,
                           n_bits,
                           signed)


def uniform_quantizer(tensor_data: np.ndarray,
                      n_bits: int,
                      signed: bool,
                      quantization_params: dict,
                      per_channel: bool,
                      output_channels_axis: int) -> np.ndarray:
    """
    Quantize a tensor uniformly in the given range.

    Args:
        tensor_data: Tensor values to quantize.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not (not used in uniform quantization).
        quantization_params: Dictionary of specific parameters for this quantization function.
        per_channel: Whether to use separate quantization per output channel.
        output_channels_axis: Axis of the output channel.

    Returns:
        Quantized data.
    """
    range_min = quantization_params.get(RANGE_MIN)
    range_max = quantization_params.get(RANGE_MAX)
    if range_min is None or range_max is None:
        Logger.error("'quantization range' parameters must be defined in 'quantization_params'")  # pragma: no cover

    return uniform_quantize_tensor(tensor_data, range_min, range_max, n_bits)
