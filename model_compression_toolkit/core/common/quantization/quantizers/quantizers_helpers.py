# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.constants import MIN_THRESHOLD, EPS
from model_compression_toolkit.core import common
from model_compression_toolkit.logger import Logger


def max_power_of_two(x: np.ndarray,
                     min_threshold: float = MIN_THRESHOLD) -> np.ndarray:
    """
    Compute the max power-of-two threshold for quantizing a tensor x. The threshold
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
    min_value = int(signed) * -threshold
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

    range_min = -threshold * int(signed)
    range_max = threshold - delta

    # Quantize the data between min/max of quantization range.
    return uniform_quantize_tensor(tensor_data,
                                   range_min=range_min,
                                   range_max=range_max,
                                   n_bits=n_bits)


def uniform_quantize_tensor(tensor_data: np.ndarray,
                            range_min: np.ndarray,
                            range_max: np.ndarray,
                            n_bits: int) -> np.ndarray:
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
    a, b = fix_range_to_include_zero(range_min, range_max, n_bits)

    # Compute the step size of quantized values.
    delta = (b - a) / (2 ** n_bits - 1)

    # Clip data in range
    clipped_tensor = np.clip(tensor_data, a_min=a, a_max=b)

    # Quantize the data between min/max of quantization range.
    q = delta * np.round((clipped_tensor - a) / delta) + a
    return q


def kmeans_assign_clusters(lut_values: np.ndarray,
                           query: np.ndarray) -> np.ndarray:
    """
    Assign each data value in query with its closest cluster center point.
    Args:
        lut_values: the cluster centers to assign the query values.
        query: values for which to assign cluster centers.

    Returns: A tensor of indexes to the cluster centers that where assigned to each value in
             the query tensor.

    """
    d0 = query.shape[0]
    d1 = lut_values.shape[0]
    query_ = query.repeat(d1).reshape(d0, d1)
    cluster_centers_ = lut_values.repeat(d0).reshape(d1, d0).transpose(1, 0)
    return np.argmin(np.abs(query_ - cluster_centers_), axis=1)


def int_quantization_with_threshold(data: np.ndarray,
                                    threshold: np.ndarray,
                                    n_bits: int,
                                    signed: bool = True,
                                    eps: float = EPS) -> np.ndarray:
    """
    Divides data by threshold and quantize it to integers in the quantization range (depends on signed value).

    Args:
        data: tensor data.
        threshold: threshold to divide the data.
        n_bits: number of bits that determines the quantization range.
        signed: Whether quantization range is signed or not (relevant only for histogram quantization).
        eps: Small value for numerical stability in division.

    Returns:
        Uniform Quantized tensor.

    """

    if signed:
        a_max = 2 ** (n_bits - 1) - 1
        a_min = -2 ** (n_bits - 1)
    else:
        a_max = 2 ** n_bits - 1
        a_min = 0

    return np.clip((data / (threshold + eps)) * (2 ** (n_bits - int(signed))), a_max=a_max, a_min=a_min)


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


def get_tensor_max(tensor_data: np.ndarray,
                   per_channel: bool,
                   channel_axis: int,
                   n_bits: int,
                   is_uniform_quantization=False) -> np.ndarray:
    """
    Returns the maximal value in the given tensor, or in each channel (if per_channel is True).
    If is_uniform_quantization is False, return the max value of a threshold so it's a no-clipping threshold, to
    avoid clipping the maximum value is case it is positive, because the maximum value represented is threshold-LSB

    Args:
        tensor_data: Tensor values to quantize.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_bits: number of bits the tensor will be quantized with
        is_uniform_quantization (bool): Whether the tensor will be quantized with uniform quantization (min-max)

    Returns: maximal value (or values).

    """
    if n_bits < 1:
        Logger.critical(f"Parameter n_bits must be positive; however 'n_bits'={n_bits} was provided.")
    if is_uniform_quantization:
        expansion_factor = 1.0
    elif n_bits == 1:
        expansion_factor = 0.0
    else:
        expansion_factor = np.power(2.0, n_bits - 1) / (np.power(2.0, n_bits - 1) - 1)
    if per_channel:
        output_shape = get_output_shape(tensor_data.shape, channel_axis)
        reshaped_tensor_data = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
        tensor_max = np.reshape(np.max(reshaped_tensor_data, axis=-1), output_shape)
        tensor_min = np.reshape(np.min(reshaped_tensor_data, axis=-1), output_shape)
        tensor_max = np.maximum(-tensor_min, tensor_max * expansion_factor)
    else:
        tensor_max = np.maximum(-np.min(tensor_data), np.max(tensor_data) * expansion_factor)
    return tensor_max


def get_tensor_min(tensor_data: np.ndarray,
                   per_channel: bool,
                   channel_axis: int) -> np.ndarray:
    """
    Returns the minimal value in the given tensor, or in each channel (if per_channel is True).
    Args:
        tensor_data: Tensor values to quantize.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.

    Returns: minimal value (or values).

    """
    if per_channel:
        output_shape = get_output_shape(tensor_data.shape, channel_axis)
        tensor_data = reshape_tensor_for_per_channel_search(tensor_data, channel_axis)
        tensor_min = np.reshape(np.min(tensor_data, axis=-1), output_shape)
    else:
        tensor_min = np.min(tensor_data)
    return tensor_min


def reshape_tensor_for_per_channel_search(tensor_data: np.ndarray, channel_axis: int) -> np.ndarray:
    """
    Reshapes the given data to be compatible for search of quantization parameters for per-channel quantization.
    Args:
        tensor_data: Tensor values to quantize.
        channel_axis: Output channel index.

    Returns: A reshaped tensor.

    """
    # rearrange the shape indices for transposing the tensor
    shape_index = [channel_axis, *[i for i in range(len(tensor_data.shape)) if i is not channel_axis]]
    # New shape of the tensor after transposing it and reshape it
    new_shape = [tensor_data.shape[channel_axis], -1]
    tensor_data_t = np.transpose(tensor_data, shape_index)
    tensor_data_r = np.reshape(tensor_data_t, new_shape)
    return tensor_data_r


def fix_range_to_include_zero(range_min: np.ndarray,
                              range_max: np.ndarray,
                              n_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusting the quantization range to include representation of 0.0 in the quantization grid.
    If quantization per-channel, then range_min and range_max should be tensors in the specific shape that allows
    quantization along the channel_axis.

    Args:
        range_min: min bound of the quantization range (before adjustment).
        range_max: max bound of the quantization range (before adjustment).
        n_bits: Number of bits to quantize the tensor.

    Returns: adjusted quantization range
    """
    min_positive = range_min > 0
    max_negative = range_max < 0

    scale = (range_max - range_min) / (2 ** n_bits - 1)

    min_range_adj = scale * np.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    mid_range = np.logical_and(np.logical_not(min_positive), np.logical_not(max_negative))
    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max
    return min_range_adj, max_range_adj


def get_output_shape(tensor_shape, channel_axis):
    """
    Returns shape vector with the number of channels in the given channel_axis location and 1 at all other locations.
    Args:
        tensor_shape: A shape vector of a tensor.
        channel_axis: Output channel index.

    Returns: A shape vector of a tensor.

    """
    return [-1 if i is channel_axis else 1 for i in range(len(tensor_shape))]


def get_range_bounds(tensor_min, tensor_max):
    """
    Gets bounds on the quantization range limits for the minimization process.
    Calculates the bounds in a way that would leave a gap between the possible optimized values
    and the tensor min-max values.

    Args:
        tensor_min: min value of a tensor.
        tensor_max: max value of a tensor.

    Returns: An array with (lbound, ubound) pairs on the quantization range limit values.

    """
    # choosing bounds that have some gap from the original tensor min/max values.
    l_bound = tensor_min / 2 if tensor_min > 0 else tensor_min * 2
    u_bound = tensor_max * 2 if tensor_max > 0 else tensor_min / 2
    return [(l_bound, u_bound), (l_bound, u_bound)]


def get_threshold_bounds(min_threshold, max_threshold):
    """
    Gets bounds on the threshold for the minimization process.
    Calculates the bounds in a way that would leave a gap between the possible optimized threshold
    and the tensor max values. We use min_threshold as lower-bound to prevent the selected threshold
    from being zero or negative.

    Args:
        min_threshold: minimal threshold to use if threshold is too small (not used for this method).
        max_threshold: maximal threshold to be used in quantization.

    Returns: An array with a pair of (lbound, ubound) on the quantization threshold limit values.

    """
    max_threshold = max(min_threshold, max_threshold)
    return [(min_threshold, 2 * max_threshold)]
