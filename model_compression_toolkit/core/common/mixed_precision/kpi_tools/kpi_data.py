# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable, Any
import numpy as np

from model_compression_toolkit import FrameworkInfo, KPI, CoreConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.constants import FLOAT_BITWIDTH
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.runner import read_model_to_graph, get_finalized_graph
from model_compression_toolkit.core.common.logger import Logger


def compute_kpi_data(in_model: Any,
                     representative_data_gen: Callable,
                     core_config: CoreConfig,
                     tpc: TargetPlatformCapabilities,
                     fw_info: FrameworkInfo,
                     fw_impl: FrameworkImplementation) -> KPI:
    """
    Compute KPI information that can be relevant for defining target KPI for mixed precision search.
    Calculates maximal activation tensor, sum of weights' parameters and total (sum of both).

    Args:
        in_model:  Model to build graph from (the model that intended to be quantized).
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        tpc: TargetPlatformCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A KPI object with the results.

    """

    graph = read_model_to_graph(in_model,
                                representative_data_gen,
                                tpc,
                                fw_info,
                                fw_impl)

    transformed_graph = get_finalized_graph(graph,
                                            tpc,
                                            core_config.quantization_config,
                                            fw_info,
                                            tb_w=None,
                                            fw_impl=fw_impl,
                                            mixed_precision_enable=core_config.mixed_precision_enable)

    # Compute parameters sum
    weights_params = compute_nodes_weights_params(graph=transformed_graph, fw_info=fw_info)
    total_weights_params = 0 if len(weights_params) == 0 else sum(weights_params)

    # Compute max activation tensor
    activation_output_sizes = compute_activation_output_sizes(graph=transformed_graph)
    max_activation_tensor_size = 0 if len(activation_output_sizes) == 0 else max(activation_output_sizes)

    # Compute total kpi - parameters sum + max activation tensor
    total_size = total_weights_params + max_activation_tensor_size

    # Compute BOPS kpi - total count of bit-operations for all configurable layers with kernel
    bops_count = compute_total_bops(graph=transformed_graph, fw_info=fw_info, fw_impl=fw_impl)
    bops_count = np.inf if len(bops_count) == 0 else sum(bops_count)

    return KPI(weights_memory=total_weights_params,
               activation_memory=max_activation_tensor_size,
               total_memory=total_size,
               bops=bops_count)


def compute_nodes_weights_params(graph: Graph, fw_info: FrameworkInfo) -> np.ndarray:
    """
    Computes a vector with the respective weights' parameters size for each node.

    Args:
        graph: Finalized Graph object.
        fw_info: FrameworkInfo object about the specific framework
            (e.g., attributes of different layers' weights to quantize).

    Returns: A vector of node's weights memory sizes.

    """

    weights_params = []
    for n in graph.nodes:
        if n.has_weights_quantization_enabled_candidate():
            node_num_weights_params = 0
            for attr in fw_info.get_kernel_op_attributes(n.type):
                if attr is not None:
                    node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]

            weights_params.append(node_num_weights_params)

    return np.array(weights_params)


def compute_activation_output_sizes(graph: Graph) -> np.ndarray:
    """
    Computes a vector with the respective output tensor size for each node.

    Args:
        graph: Finalized Graph object.

    Returns: A vector of node's activation output size.

    """

    activation_outputs = []
    # Go over all nodes that have configurable activation.
    for n in graph.nodes:
        if n.has_activation_quantization_enabled_candidate():
            node_output_size = n.get_total_output_params()
            activation_outputs.append(node_output_size)

    return np.array(activation_outputs)


def compute_total_bops(graph: Graph, fw_info: FrameworkInfo, fw_impl: FrameworkImplementation) -> np.ndarray:
    """
    Computes a vector with the respective Bit-operations count for each configurable node that includes MAC operations.
    The computation assumes that the graph is a representation of a float model, thus, BOPs computation uses 32-bit.

    Args:
        graph: Finalized Graph object.
        fw_info: FrameworkInfo object about the specific framework
            (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A vector of nodes' Bit-operations count.

    """

    bops = []

    # Go over all configurable nodes that have kernels.
    for n in graph.get_topo_sorted_nodes():
        if n.has_weights_to_quantize(fw_info):
            # If node doesn't have weights then its MAC count is 0, and we shouldn't consider it in the BOPS count.
            incoming_edges = graph.incoming_edges(n, sort_by_attr=EDGE_SINK_INDEX)
            if len(incoming_edges) != 1:
                Logger.warning(f"Can't compute BOPS metric for node {n.name} with multiple inputs.")

            node_mac = fw_impl.get_node_mac_operations(n, fw_info)

            node_bops = (FLOAT_BITWIDTH ** 2) * node_mac
            bops.append(node_bops)

    return np.array(bops)
