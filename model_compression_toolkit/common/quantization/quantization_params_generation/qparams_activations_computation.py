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

from model_compression_toolkit.common import BaseNode, Graph
from model_compression_toolkit.common.quantization import quantization_params_generation


def get_activations_qparams(n: BaseNode,
                            graph: Graph) -> Tuple[Dict[str, float], bool]:
    """
    Compute the activations params for a given node in a graph according to a params function.

    Args:
        n: Node to compute its' activations threshold.
        graph: Graph the node is in.

    Returns:
        Tuple of the computed quantization params and sign for the node's activations quantization.
    """
    out_stats_container = graph.get_out_stats_collector(n)
    bins_values, bins_counts = None, None

    # If the statistics container collected the histogram, we start by filtering outliers using z threshold
    # filtering, and then computing the threshold based on the filtered histogram.
    if out_stats_container.collect_histogram:
        bins_values, bins_counts = out_stats_container.hc.get_histogram()
        bins_counts = quantization_params_generation.z_score_filter(n.activation_quantization_cfg.z_threshold,
                                                                    bins_values,
                                                                    bins_counts)
    min_value, max_value = out_stats_container.get_min_max_values()

    if out_stats_container.use_min_max:
        signed = min_value < 0
    else:
        signed = np.any(bins_values < 0)

    activation_params = n.activation_quantization_cfg.activation_quantization_params_fn(bins_values,
                                                                                        bins_counts,
                                                                                        n.activation_quantization_cfg.l_p_value,
                                                                                        n.activation_quantization_cfg.activation_n_bits,
                                                                                        min_value,
                                                                                        max_value,
                                                                                        min_threshold=n.activation_quantization_cfg.min_threshold)

    return activation_params, signed