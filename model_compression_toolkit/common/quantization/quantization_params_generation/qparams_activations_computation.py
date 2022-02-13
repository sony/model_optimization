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
import numpy as np
from typing import Tuple, Dict

from model_compression_toolkit import QuantizationMethod
from model_compression_toolkit.common import BaseNode, Graph
from model_compression_toolkit.common.constants import SIGNED
from model_compression_toolkit.common.quantization import quantization_params_generation


def get_activations_qparams(n: BaseNode,
                            graph: Graph) -> Dict[str, float]:
    """
    Compute the activations params for a given node in a graph according to a params function.

    Args:
        n: Node to compute its' activations threshold.
        graph: Graph the node is in.

    Returns:
        The computed activation quantization params.
    """

    out_stats_container = graph.get_out_stats_collector(n)
    bins_values, bins_counts = None, None

    # If the statistics container collected the histogram, we start by filtering outliers using z threshold
    # filtering, and then computing the threshold based on the filtered histogram.
    if out_stats_container.require_collection():
        bins_values, bins_counts = out_stats_container.hc.get_histogram()
        bins_counts = quantization_params_generation.z_score_filter(n.activation_quantization_cfg.z_threshold,
                                                                    bins_values,
                                                                    bins_counts)
    min_value, max_value = out_stats_container.get_min_max_values()

    if n.prior_info.is_output_bounded():
        signed = min_value < 0
    else:
        signed = np.any(bins_values < 0)

    if n.prior_info.is_output_bounded():
        if n.activation_quantization_cfg.activation_quantization_method == QuantizationMethod.POWER_OF_TWO:
            n.activation_quantization_cfg.activation_quantization_params_fn = \
                quantization_params_generation.no_clipping_selection_min_max
        elif n.activation_quantization_cfg.activation_quantization_method == QuantizationMethod.SYMMETRIC:
            n.activation_quantization_cfg.activation_quantization_params_fn = \
                quantization_params_generation.symmetric_no_clipping_selection_min_max
        elif n.activation_quantization_cfg.activation_quantization_method == QuantizationMethod.UNIFORM:
            n.activation_quantization_cfg.activation_quantization_params_fn = \
                quantization_params_generation.uniform_no_clipping_selection_min_max

    activation_params = n.activation_quantization_cfg.activation_quantization_params_fn(bins_values,
                                                                                        bins_counts,
                                                                                        n.activation_quantization_cfg.l_p_value,
                                                                                        n.activation_quantization_cfg.activation_n_bits,
                                                                                        min_value,
                                                                                        max_value,
                                                                                        min_threshold=n.activation_quantization_cfg.min_threshold,
                                                                                        quant_error_method=n.activation_quantization_cfg.activation_error_method)
    activation_params.update({SIGNED: signed})

    return activation_params