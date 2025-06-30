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
from functools import partial
from typing import Dict, Any, Tuple, Callable, TYPE_CHECKING

import numpy as np
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.constants import NUM_QPARAM_HESSIAN_SAMPLES
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.quantization.quantization_params_generation import \
    power_of_two_selection_tensor, lut_kmeans_tensor, symmetric_selection_tensor, uniform_selection_tensor
from model_compression_toolkit.logger import Logger

if TYPE_CHECKING:
    from model_compression_toolkit.core.common.quantization.node_quantization_config import WeightsAttrQuantizationConfig


def compute_weights_qparams(weights_attr_values: np.ndarray,
                            attr_quant_config: 'WeightsAttrQuantizationConfig',
                            output_channels_axis: int,
                            min_threshold: float,
                            node=None,
                            hessian_info_service: HessianInfoService = None,
                            num_hessian_samples: int = NUM_QPARAM_HESSIAN_SAMPLES) -> Tuple[Dict[Any, Any], int]:
    """
    Compute thresholds to quantize a kernel according to a NodeWeightsQuantizationConfig
    instance.

    Args:
        weights_attr_values: Weights attribute parameter to compute the quantization thresholds for.
        attr_quant_config: A specific weights attribute quantization configuration to get its params.
        output_channels_axis: Index of the kernel output channels dimension.
        min_threshold: Minimal threshold to use if threshold is too small.
        node: The node for which the quantization error is computed (used only with HMSE error method).
        hessian_info_service: HessianInfoService object for retrieving Hessian-based scores (used only with HMSE error method).
        num_hessian_samples: Number of samples to approximate Hessian-based scores on (used only with HMSE error method).

    Returns:
        A dictionary with the quantization threshold of the kernel.
        Selected quantization channel axis.
    """
    params_fn = _get_weights_quantization_params_fn(attr_quant_config.weights_quantization_method)
    weights_params, output_channels_axis = params_fn(
        weights_attr_values,
        p=attr_quant_config.l_p_value,
        n_bits=attr_quant_config.weights_n_bits,
        per_channel=attr_quant_config.weights_per_channel_threshold,
        channel_axis=output_channels_axis,
        min_threshold=min_threshold,
        quant_error_method=attr_quant_config.weights_error_method,
        node=node,
        hessian_info_service=hessian_info_service,
        num_hessian_samples=num_hessian_samples)

    return weights_params, output_channels_axis


_weights_quant_params_fns = {
    QuantizationMethod.POWER_OF_TWO: power_of_two_selection_tensor,
    QuantizationMethod.SYMMETRIC: symmetric_selection_tensor,
    QuantizationMethod.UNIFORM: uniform_selection_tensor,
    QuantizationMethod.LUT_POT_QUANTIZER: partial(lut_kmeans_tensor, is_symmetric=False),
    QuantizationMethod.LUT_SYM_QUANTIZER: partial(lut_kmeans_tensor, is_symmetric=True)
}


def _get_weights_quantization_params_fn(weights_quantization_method: QuantizationMethod) -> Callable:
    """
    Generate a function for finding weights quantization parameters.

    Args:
        weights_quantization_method: Which quantization method to use for weights.
    Returns:
        A function to find the quantization parameters.

    """
    params_fn = _weights_quant_params_fns.get(weights_quantization_method)
    if not params_fn:
        Logger.critical(
            f"No parameter function found for the specified quantization method: {weights_quantization_method}")  # pragma: no cover
    return params_fn
