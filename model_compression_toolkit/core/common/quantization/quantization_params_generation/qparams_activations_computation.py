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
import numpy as np
from typing import Dict, Union, Optional, Tuple, Callable

from mct_quantizers import QuantizationMethod

import model_compression_toolkit.core.common.quantization.quantization_params_generation as qpg
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationErrorMethod


def compute_activation_qparams(activation_quant_cfg: NodeActivationQuantizationConfig,
                               node_prior_info: NodePriorInfo,
                               out_stats_container: BaseStatsCollector) -> Dict[str, Union[np.ndarray, float, bool]]:
    """
    Compute the activations params for a given node in a graph according to a params function.

    Args:
        activation_quant_cfg: node's activation quantization configuration.
        node_prior_info: Prior info collected for the node that is being quantized.
        out_stats_container: Tensor containing output statistics of the node.

    Returns:
        The computed activation quantization params.
    """
    activation_quantization_params_fn = _get_activation_quantization_params_fn(
        activation_quant_cfg.activation_quantization_method, no_clipping=node_prior_info.is_output_bounded())

    # Extract and filter histogram data from the statistics container.
    bins_values, bins_counts = _get_histogram_data(activation_quant_cfg, out_stats_container)

    # Retrieve the minimum and maximum values from the statistics container.
    min_value, max_value = out_stats_container.get_min_max_values()

    # Determine if the activations should be considered signed.
    signed = _determine_signedness(activation_quant_cfg, node_prior_info, min_value, bins_values, bins_counts)

    # Compute and return the activation quantization parameters.
    return activation_quantization_params_fn(
        bins_values,
        bins_counts,
        activation_quant_cfg.l_p_value,
        activation_quant_cfg.activation_n_bits,
        min_value,
        max_value,
        min_threshold=activation_quant_cfg.min_threshold,
        quant_error_method=activation_quant_cfg.activation_error_method,
        is_signed=signed
    )


def _get_histogram_data(
    activation_quant_cfg: NodeActivationQuantizationConfig,
    out_stats_container: BaseStatsCollector
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract and filter the histogram data from the statistics container.

    Args:
        activation_quant_cfg: Node's activation quantization configuration.
        out_stats_container: Statistics container with histogram data.

    Returns:
        A tuple containing the filtered bins_values and bins_counts.
    """
    bins_values, bins_counts = None, None
    # If the statistics container collected the histogram, we start by filtering outliers using z threshold
    # filtering, and then computing the threshold based on the filtered histogram.
    if out_stats_container.require_collection():
        if activation_quant_cfg.activation_error_method == QuantizationErrorMethod.HMSE:
            bins_values, bins_counts = out_stats_container.weighted_hc.get_histogram()
        else:
            bins_values, bins_counts = out_stats_container.hc.get_histogram()
        bins_counts = qpg.z_score_filter(
            activation_quant_cfg.z_threshold,
            bins_values,
            bins_counts
        )
    return bins_values, bins_counts


def _determine_signedness(
    activation_quant_cfg: NodeActivationQuantizationConfig,
    nodes_prior_info: NodePriorInfo,
    min_value: float,
    bins_values: Optional[np.ndarray],
    bins_counts: Optional[np.ndarray]
) -> bool:
    """
    Determine if the activations should be considered signed based on the quantization configuration,
    node prior information, and histogram statistics.

    Args:
        activation_quant_cfg: Node's activation quantization configuration.
        nodes_prior_info: Prior info collected for the node that is being quantized.
        min_value: Minimum value from the statistics container.
        bins_values: Numpy array of histogram bin values.
        bins_counts: Numpy array of histogram bin counts.

    Returns:
        A boolean indicating if the activations are signed.
    """
    if activation_quant_cfg.signedness in [Signedness.SIGNED, Signedness.UNSIGNED]:
        return activation_quant_cfg.signedness == Signedness.SIGNED

    if nodes_prior_info.is_output_bounded():
        return min_value < 0

    return np.any(bins_values[:-1][bins_counts > 0] < 0)


_activation_quant_params_fns = {
    QuantizationMethod.POWER_OF_TWO: qpg.power_of_two_selection_histogram,
    QuantizationMethod.SYMMETRIC: qpg.symmetric_selection_histogram,
    QuantizationMethod.UNIFORM: qpg.uniform_selection_histogram,
    QuantizationMethod.LUT_POT_QUANTIZER: qpg.lut_kmeans_histogram
}
_activation_no_clipping_quant_params_fns = {
    QuantizationMethod.POWER_OF_TWO: qpg.power_of_two_no_clipping_selection_min_max,
    QuantizationMethod.SYMMETRIC: qpg.symmetric_no_clipping_selection_min_max,
    QuantizationMethod.UNIFORM: qpg.uniform_no_clipping_selection_min_max,
    QuantizationMethod.LUT_POT_QUANTIZER: qpg.lut_kmeans_histogram
}


def _get_activation_quantization_params_fn(activation_quantization_method: QuantizationMethod,
                                           no_clipping: bool) -> Callable:
    """
    Generate a function for finding activation quantization parameters.

    Args:
        activation_quantization_method: Which quantization method to use for activations.
        no_clipping: Whether to use the no-clipping version of the quantizer (if available).

    Returns:
        A function to find the quantization parameters.
    """
    if no_clipping:
        params_fn = _activation_no_clipping_quant_params_fns.get(activation_quantization_method)
    else:
        params_fn = _activation_quant_params_fns.get(activation_quantization_method)
    if params_fn is None:
        raise ValueError(f"No parameter function found for the specified quantization method: "
                         "{activation_quantization_method}")  # pragma: no cover
    return params_fn
