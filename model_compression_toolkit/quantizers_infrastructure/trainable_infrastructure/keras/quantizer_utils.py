# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import tensorflow as tf


def int_quantization_with_threshold(data: tf.Tensor,
                                    n_bits: int,
                                    signed: bool,
                                    threshold: np.ndarray,
                                    eps: float) -> tf.Tensor:
    """
    Divides data by threshold and quantize it to integers in the quantization range (depends on signed value).

    Args:
        data: tensor data.
        n_bits: number of bits that determines the quantization range.
        signed: Whether the quantization is signed or not.
        threshold: threshold for quantization.
        eps: Small value for numerical stability in division.

    Returns:
        Uniform Quantized tensor.

    """

    if signed:
        clip_max = 2 ** (n_bits - 1) - 1
        clip_min = -2 ** (n_bits - 1)
    else:
        clip_max = 2 ** n_bits - 1
        clip_min = 0

    return tf.clip_by_value((data / (threshold + eps)) * (2 ** (n_bits - int(signed))),
                            clip_value_max=clip_max, clip_value_min=clip_min)
