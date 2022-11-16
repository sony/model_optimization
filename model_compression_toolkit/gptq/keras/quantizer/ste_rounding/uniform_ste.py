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
import tensorflow as tf

from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils


def rounding_uniform_quantizer(tensor_data: tf.Tensor,
                               range_min: tf.Tensor,
                               range_max: tf.Tensor,
                               n_bits: int) -> tf.Tensor:
    """
    Quantize a tensor according to given range (min, max) and number of bits.

    Args:
        tensor_data: Tensor values to quantize.
        range_min: minimum bound of the range for quantization (or array of min values per channel).
        range_max: maximum bound of the range for quantization (or array of max values per channel).
        n_bits: Number of bits to quantize the tensor.

    Returns:
        Quantized data.
    """
    # adjusts the quantization rage so the quantization grid include zero.
    a, b = qutils.fix_range_to_include_zero(range_min, range_max, n_bits)

    # Compute the step size of quantized values.
    delta = (b - a) / (2 ** n_bits - 1)

    input_tensor_int = qutils.ste_round((tensor_data - a) / delta)  # Apply rounding

    # Clip data in range
    clipped_tensor = qutils.ste_clip(input_tensor_int, min_val=0, max_val=2 ** n_bits - 1)

    # Quantize the data between min/max of quantization range.
    q = delta * clipped_tensor + a
    return q
