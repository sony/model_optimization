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
from typing import Tuple


def adjust_range_to_include_zero(range_min: np.ndarray,
                                 range_max: np.ndarray,
                                 n_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusting the quantization range to include representation of 0.0 in the quantization grid.
    For per_channel quantization range_min\range_max should be tensors in the specific shape that allows
    quantization along the channel_axis.

    Args:
        range_min: min bound of the quantization range (before adjustment).
        range_max: max bound of the quantization range (before adjustment).
        n_bits: Number of bits to quantize the tensor.

    Returns: adjusted quantization range
    """
    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * np.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    min_positive = range_min > 0
    max_negative = range_max < 0
    mid_range = np.logical_and(np.logical_not(min_positive), np.logical_not(max_negative))

    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max

    # Make sure min_range_adj < 0 and max_range_adj > 0 to avoid small numeric error
    min_range_adj = np.minimum(min_range_adj, 0)
    max_range_adj = np.maximum(max_range_adj, 0)

    return min_range_adj, max_range_adj
