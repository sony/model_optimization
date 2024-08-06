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
from typing import Dict, Union

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod, Signedness
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.core.common.quantization import quantization_params_generation
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig


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

    bins_values, bins_counts = None, None

    # If the statistics container collected the histogram, we start by filtering outliers using z threshold
    # filtering, and then computing the threshold based on the filtered histogram.
    if out_stats_container.require_collection():
        bins_values, bins_counts = out_stats_container.hc.get_histogram()
        bins_counts = quantization_params_generation.z_score_filter(activation_quant_cfg.z_threshold,
                                                                    bins_values,
                                                                    bins_counts)
    min_value, max_value = out_stats_container.get_min_max_values()

    if activation_quant_cfg.signedness in [Signedness.SIGNED, Signedness.UNSIGNED]:
        signed = activation_quant_cfg.signedness == Signedness.SIGNED
    elif nodes_prior_info.is_output_bounded():
        signed = min_value < 0
    else:
        signed = np.any(bins_values[:-1][bins_counts > 0] < 0)

    if nodes_prior_info.is_output_bounded():
        if activation_quant_cfg.activation_quantization_method == QuantizationMethod.POWER_OF_TWO:
            activation_quant_cfg.activation_quantization_params_fn = \
                quantization_params_generation.power_of_two_no_clipping_selection_min_max
        elif activation_quant_cfg.activation_quantization_method == QuantizationMethod.SYMMETRIC:
            activation_quant_cfg.activation_quantization_params_fn = \
                quantization_params_generation.symmetric_no_clipping_selection_min_max
        elif activation_quant_cfg.activation_quantization_method == QuantizationMethod.UNIFORM:
            activation_quant_cfg.activation_quantization_params_fn = \
                quantization_params_generation.uniform_no_clipping_selection_min_max

    return activation_quant_cfg.activation_quantization_params_fn(bins_values,
                                                                  bins_counts,
                                                                  activation_quant_cfg.l_p_value,
                                                                  activation_quant_cfg.activation_n_bits,
                                                                  min_value,
                                                                  max_value,
                                                                  min_threshold=activation_quant_cfg.min_threshold,
                                                                  quant_error_method=activation_quant_cfg.activation_error_method,
                                                                  is_signed=signed)
