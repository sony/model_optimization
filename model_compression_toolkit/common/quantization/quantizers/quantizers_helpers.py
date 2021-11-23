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


from typing import Tuple, List
import numpy as np

from model_compression_toolkit.common.constants import MIN_THRESHOLD, EPS


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
                           query: np.ndarray) -> np.ndarray:
    """
    Assign each data value in query with its closest cluster center point.
    Args:
        cluster_centers: the cluster centers to assign the query values.
        query: values for which to assign cluster centers.

    Returns: A tensor of indexes to the cluster centers that where assigned to each value in
             the query tensor.

    """
    d0 = query.shape[0]
    d1 = cluster_centers.shape[0]
    query_ = query.repeat(d1).reshape(d0, d1)
    cluster_centers_ = cluster_centers.repeat(d0).reshape(d1, d0).transpose(1, 0)
    return np.argmin(np.abs(query_ - cluster_centers_), axis=1)


def int_quantization_with_scale(data: np.ndarray,
                                scale: np.ndarray,
                                n_bits: int,
                                eps: float = EPS) -> np.ndarray:
    """
    Divides data by scale and quantizes it to integers in the range [2 ** (n_bits - 1) - 1, -2 ** (n_bits - 1)]
    Args:
        data: tensor data.
        scale: scale to divide the data.
        n_bits: number of bits that determines the quantization range.

    Returns:
        Quantized tensor.

    """
    return np.clip((data) / (scale + eps) * 2 ** (n_bits - 1),
                   a_max=2 ** (n_bits - 1) - 1, a_min=-2 ** (n_bits - 1))


def get_quantized_tensor(centers: np.ndarray,
                                scale: np.ndarray,
                                n_bits: int) -> np.ndarray:
    """
    Divides data by scale and quantizes it to integers in the range [2 ** (n_bits - 1) - 1, -2 ** (n_bits - 1)]
    Args:
        centers: centers points.
        scale: scale to divide the data.
        n_bits: number of bits that determines the quantization range.

    Returns:
        Quantized tensor.

    """
    return (np.round(centers) / (2 ** (n_bits - 1))) * scale