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

from model_compression_toolkit.common.quantization import quantization_params_generation
from model_compression_toolkit.common.quantization.quantization_config import ThresholdSelectionMethod, \
    QuantizationMethod
from model_compression_toolkit.common.quantization.quantization_params_generation.kmeans_params import kmeans_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation.lut_kmeans_params import lut_kmeans_tensor


def get_activation_quantization_params_fn(activation_quantization_method: QuantizationMethod,
                                          activation_threshold_method: ThresholdSelectionMethod,
                                          use_min_max: bool) -> Callable:
    """
    Generate a function for finding activation quantization threshold.

    Args:
        activation_quantization_method: Which quantization method to use for activations.
        activation_threshold_method: Method for finding the threshold for activation quantization.

    Returns:
        A function to find the quantization threshold.

    """
    if activation_quantization_method == QuantizationMethod.POWER_OF_TWO:
        # Use min/max as the threshold if we use NOCLIPPING
        if use_min_max or activation_threshold_method == ThresholdSelectionMethod.NOCLIPPING:
            params_fn = quantization_params_generation.no_clipping_selection_min_max
        # Use MSE to search_methods for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.MSE:
            params_fn = quantization_params_generation.mse_selection_histogram
        # Use MAE to search_methods for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.MAE:
            params_fn = quantization_params_generation.mae_selection_histogram
        # Use Lp distance to search_methods for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.LP:
            params_fn = quantization_params_generation.lp_selection_histogram
        # Use KL-divergence to search_methods for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.KL:
            params_fn = quantization_params_generation.kl_selection_histogram
        else:
            params_fn = None
    else:
        raise Exception(
            f'No params function for the configuration of quantization method {activation_quantization_method} and '
            f'threshold selection method {activation_threshold_method}')
    return params_fn


def get_weights_quantization_params_fn(weights_quantization_method: QuantizationMethod,
                                       weights_threshold_method: ThresholdSelectionMethod) -> Callable:
    """
    Generate a function for finding weights quantization threshold.

    Args:
        weights_quantization_method: Which quantization method to use for weights.
        weights_threshold_method: Method for finding the threshold for weight quantization.

    Returns:
        A function to find the quantization threshold.

    """
    if weights_quantization_method == QuantizationMethod.POWER_OF_TWO:
        if weights_threshold_method == ThresholdSelectionMethod.NOCLIPPING:
            params_fn = quantization_params_generation.no_clipping_selection_tensor
        # Use MSE to search_methods for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.MSE:
            params_fn = quantization_params_generation.mse_selection_tensor
        # Use MAE to search_methods for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.MAE:
            params_fn = quantization_params_generation.mae_selection_tensor
        # Use KL-divergence to search_methods for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.KL:
            params_fn = quantization_params_generation.kl_selection_tensor
        # Use Lp distance to search_methods for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.LP:
            params_fn = quantization_params_generation.lp_selection_tensor
        else:
            params_fn = None
    elif weights_quantization_method == QuantizationMethod.KMEANS:
        params_fn = kmeans_tensor
    elif weights_quantization_method == QuantizationMethod.LUT_QUANTIZER:
        params_fn = lut_kmeans_tensor
    else:
        raise Exception(
            f'No params function for the configuration of quantization method {weights_quantization_method} and '
            f'threshold selection method {weights_threshold_method}')
    return params_fn
