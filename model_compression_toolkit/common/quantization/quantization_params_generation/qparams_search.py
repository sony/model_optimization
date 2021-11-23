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

from collections import Callable
from typing import Any

import numpy as np

from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import quantize_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation.no_clipping import no_clipping_selection_tensor, \
    no_clipping_selection_histogram


def qparams_selection_tensor_search(error_function: Callable,
                                    tensor_data: np.ndarray,
                                    n_bits: int,
                                    per_channel: bool = False,
                                    channel_axis: int = 1,
                                    n_iter: int = 10,
                                    min_threshold=MIN_THRESHOLD) -> Any:
    """
    Search for an optimal threshold to quantize a tensor.
    The search_methods starts with the constrained no-clipping threshold the tensor has, and continues with
    n_iter another smaller constrained thresholds. For each candidate threshold, an error is computed
    based on the passed error function, and the threshold which yields the minimal error is selected
    and returned.

    Args:
        error_function: Function to compute the error between the original and quantized tensors.
        tensor_data: Numpy array with tensor's content.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the tensor should be quantized per-channel or per-tensor.
        channel_axis: Index of output channels dimension.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Optimal constrained threshold to quantize the tensor.

    """

    signed = np.any(tensor_data < 0)  # check if tensor is singed
    output_shape = [-1 if i is channel_axis else 1 for i in range(len(tensor_data.shape))]

    # First threshold to check is the constrained threshold based on the tensor's maximal value.
    threshold = 2 * no_clipping_selection_tensor(tensor_data,
                                                 0,
                                                 n_bits,
                                                 per_channel,
                                                 channel_axis,
                                                 min_threshold=min_threshold)[THRESHOLD]

    # If the threshold is computed per-channel, we rearrange the tensor such that each sub-tensor
    # is flattened, and we iterate over each one of them when searching for the threshold.
    if per_channel:
        # rearrange the shape indices for transposing the tensor
        shape_index = [channel_axis, *[i for i in range(len(tensor_data.shape)) if i is not channel_axis]]
        # New shape of the tensor after transposing it and reshape it
        new_shape = [tensor_data.shape[channel_axis], -1]
        tensor_data_t = np.transpose(tensor_data, shape_index)
        tensor_data_r = np.reshape(tensor_data_t, new_shape)

    error_list = []  # init an empty error list
    # On each iteration a new constrained threshold which equal to half of the previous tested threshold
    # is used for quantizing the tensor and computing the error. The error is appended to an error list, which
    # eventually used to select the threshold with the minimal error.
    for i in range(n_iter):
        if per_channel:
            per_channel_error = []
            for j in range(tensor_data_r.shape[0]):  # iterate all channels of the tensor.
                qt = quantize_tensor(tensor_data_r[j, :], threshold.flatten()[j] / (2 ** i), n_bits, signed)
                error = error_function(qt, tensor_data_r[j, :], threshold=threshold.flatten()[j] / (2 ** i))
                per_channel_error.append(error)
            error_list.append(np.asarray(per_channel_error))
        else:  # quantize per-tensor
            qt = quantize_tensor(tensor_data, threshold / (2 ** i), n_bits, signed)
            error = error_function(qt, tensor_data, threshold=threshold / (2 ** i))
            error_list.append(error)

    # Take the index of the minimal error, and use it compute the threshold which yielded it.
    i = np.argmin(np.stack(error_list, axis=-1), axis=-1)

    return np.maximum(np.reshape(threshold.flatten() / np.power(2, i), output_shape), min_threshold)


def qparams_selection_histogram_search(error_function: Callable,
                                       bins: np.ndarray,
                                       counts: np.ndarray,
                                       n_bits: int,
                                       constrained: bool = True,
                                       n_iter: int = 10,
                                       min_threshold: float = MIN_THRESHOLD):
    """
    Search for an optimal threshold to quantize a histogram of collected float values.
    The search_methods starts with the constrained no-clipping threshold by the bins' maximal value, and continues with
    n_iter another smaller constrained thresholds. For each candidate threshold, an error is computed
    based on the passed error function, and the threshold which yields the minimal error is selected
    and returned.

    Args:
        error_function: Function to compute the error between the original and quantized histograms.
        bins: Bins of the histogram to search_methods for an optimal threshold.
        counts: Number of elements in the bins to search_methods for a threshold.
        n_bits: Number of bits to quantize the tensor.
        constrained: Whether the threshold should be constrained or not.
        n_iter: Number of searching iterations.
        min_threshold: Threshold to return if the computed threshold is smaller that min_threshold.

    Returns:
        Optimal constrained threshold to quantize the tensor.

    """

    signed = np.any(bins < 0)  # Whether histogram contains negative values or not.
    threshold = (1 + int(constrained)) * no_clipping_selection_histogram(bins,
                                                                         counts,
                                                                         p=0,#dummy
                                                                         n_bits=n_bits,#dummy
                                                                         min_value=0,#dummy
                                                                         max_value=0,#dummy
                                                                         constrained=constrained,
                                                                         n_iter=n_iter, #dummy
                                                                         min_threshold=min_threshold)
    # Init a list of thresholds.
    error_list = []
    threshold_list = threshold / np.power(2, np.linspace(0, n_iter - 1, n_iter))

    # On each iteration a new constrained threshold which equal to half of the previous tested threshold
    # is used for quantizing the histogram and computing the error. The error is appended to an error list, which
    # eventually used to select the threshold with the minimal error.
    for threshold in threshold_list:
        q_bins = quantize_tensor(bins, threshold, n_bits, signed)  # compute the quantized values of the bins.
        # compute the number of elements between qantized bin values.
        q_count, _ = np.histogram(q_bins, bins=bins, weights=np.concatenate([counts.flatten(), np.asarray([0])]))
        error = error_function(q_bins, q_count, bins, counts)  # compute the error
        error_list.append(error)

    # Return the threshold with the minimal error.
    return np.maximum(threshold_list[np.argmin(error_list)], min_threshold)
