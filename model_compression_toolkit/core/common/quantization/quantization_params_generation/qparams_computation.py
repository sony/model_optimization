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
from tqdm import tqdm
from typing import List

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_activations_computation \
    import get_activations_qparams
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_weights_computation import \
    get_weights_qparams
from model_compression_toolkit.logger import Logger


def calculate_quantization_params(graph: Graph,
                                  nodes: List[BaseNode] = [],
                                  specific_nodes: bool = False):
    """
    For a graph, go over its nodes, compute quantization params (for both weights and activations according
    to the given framework info), and create and attach a NodeQuantizationConfig to each node (containing the
    computed params).
    By default, the function goes over all nodes in the graph. However, the specific_nodes flag enables
    to compute quantization params for specific nodes if the default behavior is unnecessary. For that,
    a list of nodes should be passed as well.

    Args:
        groups of layers by how they should be quantized, etc.)
        graph: Graph to compute its nodes' thresholds.
        nodes: List of nodes to compute their thresholds instead of computing it for all nodes in the graph.
        specific_nodes: Flag to compute thresholds for only specific nodes.

    """

    Logger.info(f"Running quantization parameters search. "
                f"This process might take some time, "
                f"depending on the model size and the selected quantization methods.\n")

    # Create a list of nodes to compute their thresholds
    nodes_list: List[BaseNode] = nodes if specific_nodes else graph.nodes()

    for n in tqdm(nodes_list, "Calculating quantization params"):  # iterate only nodes that we should compute their thresholds
        for candidate_qc in n.candidates_quantization_cfg:
            for attr in n.get_node_weights_attributes():
                if n.is_weights_quantization_enabled(attr):
                    # If the node's weights attribute should be quantized, we compute its quantization parameters
                    attr_cfg = candidate_qc.weights_quantization_cfg.get_attr_config(attr)
                    channels_axis = attr_cfg.weights_channels_axis
                    if channels_axis is not None:
                        output_channels_axis = channels_axis[0]
                    else:
                        output_channels_axis = None
                    weights_params = get_weights_qparams(n.get_weights_by_keys(attr),
                                                         candidate_qc.weights_quantization_cfg,
                                                         attr_cfg,
                                                         output_channels_axis)
                    attr_cfg.set_weights_quantization_param(weights_params)

            if n.is_activation_quantization_enabled():
                # If node's activations should be quantized as well, we compute its activation quantization parameters
                activation_params = get_activations_qparams(
                    activation_quant_cfg=candidate_qc.activation_quantization_cfg,
                    nodes_prior_info=n.prior_info,
                    out_stats_container=graph.get_out_stats_collector(n))
                # Create a NodeQuantizationConfig containing all quantization params and attach it to the node
                candidate_qc.activation_quantization_cfg.set_activation_quantization_param(activation_params)
