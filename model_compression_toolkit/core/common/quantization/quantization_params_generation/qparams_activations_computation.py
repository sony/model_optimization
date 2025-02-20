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
from typing import Dict, Union, Optional, Tuple

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.core.common.quantization import quantization_params_generation
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig

def get_histogram_data(
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
        bins_counts = quantization_params_generation.z_score_filter(
            activation_quant_cfg.z_threshold,
            bins_values,
            bins_counts
        )
    return bins_values, bins_counts

def determine_signedness(
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


def update_activation_quantization_params_fn(
        activation_quant_cfg: NodeActivationQuantizationConfig,
        nodes_prior_info: NodePriorInfo):
    """
    Update the activation quantization parameters function based on the quantization method
    and whether the node's output is bounded.

    Args:
        activation_quant_cfg: Node's activation quantization configuration.
        nodes_prior_info: Prior info collected for the node that is being quantized.
    """
    if nodes_prior_info.is_output_bounded():
        if activation_quant_cfg.activation_quantization_method == QuantizationMethod.POWER_OF_TWO:
            activation_quant_cfg.set_activation_quantization_params_fn(
                quantization_params_generation.power_of_two_no_clipping_selection_min_max
            )
        elif activation_quant_cfg.activation_quantization_method == QuantizationMethod.SYMMETRIC:
            activation_quant_cfg.set_activation_quantization_params_fn(
                quantization_params_generation.symmetric_no_clipping_selection_min_max
            )
        elif activation_quant_cfg.activation_quantization_method == QuantizationMethod.UNIFORM:
            activation_quant_cfg.set_activation_quantization_params_fn(
                quantization_params_generation.uniform_no_clipping_selection_min_max
            )


def get_activations_qparams(activation_quant_cfg: NodeActivationQuantizationConfig,
                            nodes_prior_info: NodePriorInfo,
                            out_stats_container: BaseStatsCollector) -> Dict[str, Union[np.ndarray, float, bool]]:
    """
    Compute the activations params for a given node in a graph according to a params function.

    Args:
        activation_quant_cfg: node's activation quantization configuration.
        nodes_prior_info: Prior info collected for the node that is being quantized.
        out_stats_container: Tensor containing output statistics of the node.

    Returns:
        The computed activation quantization params.
    """
    # Update quantization parameters function based on output bounds and quantization method.
    update_activation_quantization_params_fn(activation_quant_cfg, nodes_prior_info)

    # Extract and filter histogram data from the statistics container.
    bins_values, bins_counts = get_histogram_data(activation_quant_cfg, out_stats_container)

    # Retrieve the minimum and maximum values from the statistics container.
    min_value, max_value = out_stats_container.get_min_max_values()

    # Determine if the activations should be considered signed.
    signed = determine_signedness(
        activation_quant_cfg,
        nodes_prior_info,
        min_value,
        bins_values,
        bins_counts
    )

    # Compute and return the activation quantization parameters.
    return activation_quant_cfg.activation_quantization_params_fn(
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