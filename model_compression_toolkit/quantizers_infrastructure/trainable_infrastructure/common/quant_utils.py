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
from typing import Tuple, List


def get_threshold_reshape_shape(tensor_shape: Tuple, quant_axis: int, quant_axis_dim: int) -> List[int]:
    """
    Gets a shape that contains 1 in all axis except the quantization axis, to adjust the threshold tensor for
    per-channel quantization.

    Args:
        tensor_shape: The shape of th

        e tensor to be quantized.
        quant_axis: The axis along which the quantization happens (usually the tensor's channel axis).
        quant_axis_dim: The dimension of the quantization axis.

    Returns: A shape to reshape the threshold tensor according to.

    """
    n_axis = len(tensor_shape)
    quantization_axis = n_axis + quant_axis if quant_axis < 0 else quant_axis

    return [quant_axis_dim if i == quantization_axis else 1 for i in range(n_axis)]
