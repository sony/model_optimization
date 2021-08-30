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


from typing import Tuple, List
import numpy as np

from sony_model_optimization_package.common.constants import MIN_THRESHOLD


def power_of_two_constraint(x: np.ndarray,
                            min_threshold: float = MIN_THRESHOLD) -> np.ndarray:
    """
    Compute the power-of-two threshold for quantizing a tensor x. The threshold
    is determined by the maximal value of the tensor (or min_threshold, the greater one, if a
    minimal value needed to be enforced for the threshold calculation).

    Args:
        x: Tensor to compute its threshold.
        min_threshold: Value to use for threshold computation if the maximal value of x
        is smaller than it.

    Returns:
        A constrained threshold to use when quantizing tensor x.
    """

    return np.power(2.0, np.ceil(np.log2(np.maximum(x, min_threshold))))


def calculate_delta(threshold: np.ndarray,
                    n_bits: int = 8,
                    signed: bool = False) -> np.ndarray:
    """
    Compute the step size of quantized values given the threshold, number of bits
    and whether its signed or unsigned.

    Args:
        threshold: Threshold to compute the step size according to.
        n_bits: Number of bits to compute the step size according to.
        signed: Whether quantization range is signed or not.

    Returns:
        Step size of quantized values according to a threshold, signedness and number of bits.
    """

    return threshold / (2 ** (n_bits - int(signed)))


def calculate_min_max_values(threshold: np.ndarray,
                             n_bits: int = 8,
                             signed: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the min/max values of a quantization range according to the threshold,
    number of bits and whether its signed or unsigned.

    Args:
        threshold: Threshold of quantization range to compute its min/max values.
        n_bits: Number of bits used in the quantization.
        signed: Whether the quantization range is signed or not.

    Returns:
        Min/max values of quantization range.
    """

    delta = calculate_delta(threshold,
                            n_bits=n_bits,
                            signed=signed)

    # If unsigned: min=0, otherwise its -threshold
    min_value = -(1 - int(signed)) * threshold
    max_value = threshold - delta

    return min_value, max_value


def quantize_tensor(tensor_data: np.ndarray,
                    threshold: np.ndarray,
                    n_bits: int,
                    signed: bool) -> np.ndarray:
    """
    Quantize a tensor according to given: threshold, number of bits, and whether
    quantization range is sign or unsigned.

    Args:
        tensor_data: Tensor values to quantize.
        threshold: Threshold for quantization ranges.
        n_bits: Number of bits to quantize the tensor.
        signed: Whether the tensor contains negative values or not.

    Returns:
        Quantized data.
    """

    # Compute the step size of quantized values.
    delta = calculate_delta(threshold,
                            n_bits,
                            signed=signed)

    # Quantize the data between min/max of quantization range.
    q = delta * np.round(tensor_data / delta)
    return np.clip(q,
                   a_min=-threshold * int(signed),
                   a_max=threshold - delta)


def kmeans_assign_clusters(cluster_centers: np.ndarray,
                           query: np.ndarray) -> List[np.ndarray]:
    cluster_assignments = []
    for i in range(query.shape[0]):
        cluster_assignments.append(np.argmin(np.sum(np.abs(cluster_centers - query[i, :])**2, axis=1)))
    return cluster_assignments
