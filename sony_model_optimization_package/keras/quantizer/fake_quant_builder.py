# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================



from typing import Tuple, Callable

import tensorflow as tf
import numpy as np
from tensorflow.python.util.object_identity import Reference as TFReference

from sony_model_optimization_package.common.constants import THRESHOLD


def quantizer_min_max_calculator(threshold: np.ndarray,
                                 num_bits: int,
                                 signed: bool) -> Tuple[float, float]:
    """
    Compute quantization range's min/max values given a threshold, number of bits,
     and whether it's signed or not.

    Args:
        threshold: Threshold for quantization range values.
        num_bits: Number of bits to use for quantization.
        signed: Whether the quantization range should include negative values or not.

    Returns:
        Min and max values for quantization range.
    """

    if signed:
        delta = threshold / (2 ** (num_bits - 1))
        min_value = -threshold
    else:
        delta = threshold / (2 ** (num_bits))
        min_value = 0

    max_value = threshold - delta
    return min_value, max_value


def constraint_quantization(activation_n_bits: int,
                            activation_is_signed: bool,
                            quantization_params: dict) -> Callable:
    """
    Use a NodeQuantizationConfig to compute a quantizer min/max values, and use it to
    build and return a fake-quantization node.

    Args:
        activation_n_bits: Number of bits to use for quantization.
        activation_is_signed: Whether the quantization range should include negative values or not.
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake quantization node.
    """
    activation_threshold = quantization_params.get(THRESHOLD)
    if activation_threshold is None:
        return None

    min_value, max_value = quantizer_min_max_calculator(activation_threshold,
                                                        activation_n_bits,
                                                        activation_is_signed)

    def q(x: TFReference) -> TFReference:
        """
        Fake-quantize the input tensor x, using a tensorflow fake-quantization node.

        Args:
            x: Input tensor to quantize.

        Returns:
            The fake-quantized input tensor.
        """
        return tf.quantization.fake_quant_with_min_max_vars(x,
                                                            min=min_value,
                                                            max=max_value,
                                                            num_bits=activation_n_bits)

    return q
