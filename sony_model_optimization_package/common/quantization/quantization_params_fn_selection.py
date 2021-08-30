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

from collections import Callable

from sony_model_optimization_package.common import threshold_selection
from sony_model_optimization_package.common.quantization.quantization_config import ThresholdSelectionMethod, \
    QuantizationMethod
from sony_model_optimization_package.common.quantization.quantizers.kmeans_quantizer import kmeans_tensor
from sony_model_optimization_package.common.quantization.quantizers.lut_kmeans_quantizer import lut_kmeans_tensor


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
    if activation_quantization_method == QuantizationMethod.SYMMETRIC_UNIFORM:
        # Use min/max as the threshold if we use NOCLIPPING
        if use_min_max or activation_threshold_method == ThresholdSelectionMethod.NOCLIPPING:
            params_fn = threshold_selection.no_clipping_selection_min_max
        # Use MSE to search for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.MSE:
            params_fn = threshold_selection.mse_selection_histogram
        # Use MAE to search for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.MAE:
            params_fn = threshold_selection.mae_selection_histogram
        # Use Lp distance to search for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.LP:
            params_fn = threshold_selection.lp_selection_histogram
        # Use KL-divergence to search for the optimal threshold.
        elif activation_threshold_method == ThresholdSelectionMethod.KL:
            params_fn = threshold_selection.kl_selection_histogram
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
    if weights_quantization_method == QuantizationMethod.SYMMETRIC_UNIFORM:
        if weights_threshold_method == ThresholdSelectionMethod.NOCLIPPING:
            params_fn = threshold_selection.no_clipping_selection_tensor
        # Use MSE to search for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.MSE:
            params_fn = threshold_selection.mse_selection_tensor
        # Use MAE to search for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.MAE:
            params_fn = threshold_selection.mae_selection_tensor
        # Use KL-divergence to search for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.KL:
            params_fn = threshold_selection.kl_selection_tensor
        # Use Lp distance to search for the optimal weights thresholds.
        elif weights_threshold_method == ThresholdSelectionMethod.LP:
            params_fn = threshold_selection.lp_selection_tensor
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
