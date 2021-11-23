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

import numpy as np

from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor


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
        raise Exception("'weights_threshold' parameter must be defined in 'quantization_params'")

    return quantize_tensor(tensor_data,
                           threshold,
                           n_bits,
                           signed)
