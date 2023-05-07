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
from model_compression_toolkit.constants import BITS_TO_BYTES


def compute_quantize_tensor_memory_bytes(tensor_size: float, n_bits: int) -> float:
    """
    A utility function to compute the actual memory size of a tensor for a given bit-width.

    Args:
        tensor_size: The number of parameters in the tensor.
        n_bits: The bit-width in which the tensor values are represented.

    Returns: The size of the tensor in memory in bytes.

    """
    return tensor_size * n_bits / BITS_TO_BYTES
