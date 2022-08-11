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
from enum import Enum
from functools import partial
from typing import List

import numpy as np

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.constants import BITS_TO_BYTES, FLOAT_BITWIDTH
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode, VirtualSplitActivationNode
from model_compression_toolkit.core.common.logger import Logger


def weights_size_kpi(mp_cfg: List[int],
                     graph: Graph,
                     fw_info: FrameworkInfo,
                     fw_impl: FrameworkImplementation) -> np.ndarray:
    """
    Computes a KPIs vector with the respective weights' memory size for each weight configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with specific framework methods implementation (not used in this method).

    Returns: A vector of node's weights memory sizes.
    Note that the vector is not necessarily of the same length as the given config.

    """
    weights_memory = []

    # Go over all nodes that should be taken into consideration when computing the weights KPI.
    mp_nodes = graph.get_configurable_sorted_nodes_names()
    for n in graph.get_sorted_weights_configurable_nodes():
        node_idx = mp_nodes.index(n.name)
        node_qc = n.candidates_quantization_cfg[mp_cfg[node_idx]]
        node_nbits = node_qc.weights_quantization_cfg.weights_n_bits

        n = _get_origin_weights_node(n)

        node_num_weights_params = 0
        for attr in fw_info.get_kernel_op_attributes(n.type):
            if attr is not None:
                node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]

        node_weights_memory_in_bytes = node_num_weights_params * node_nbits / BITS_TO_BYTES
        weights_memory.append(node_weights_memory_in_bytes)

    return np.array(weights_memory)


def activation_output_size_kpi(mp_cfg: List[int],
                               graph: Graph,
                               fw_info: FrameworkInfo,
                               fw_impl: FrameworkImplementation) -> np.ndarray:
    """
    Computes a KPIs vector with the respective output memory size for each activation configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize)
            (not used in this method).
        fw_impl: FrameworkImplementation object with specific framework methods implementation(not used in this method).

    Returns: A vector of node's activation memory sizes.
    Note that the vector is not necessarily of the same length as the given config.

    """

    activation_memory = []

    # Go over all nodes that should be taken into consideration when computing the weights KPI.
    mp_nodes = graph.get_configurable_sorted_nodes_names()
    for n in graph.get_sorted_activation_configurable_nodes():
        node_idx = mp_nodes.index(n.name)
        node_qc = n.candidates_quantization_cfg[mp_cfg[node_idx]]
        node_nbits = node_qc.activation_quantization_cfg.activation_n_bits

        n = _get_origin_activation_node(n)

        node_output_size = n.get_total_output_params()
        node_activation_memory_in_bytes = node_output_size * node_nbits / BITS_TO_BYTES
        activation_memory.append(node_activation_memory_in_bytes)

    return np.array(activation_memory)


def total_weights_activation_kpi(mp_cfg: List[int],
                                 graph: Graph,
                                 fw_info: FrameworkInfo,
                                 fw_impl: FrameworkImplementation) -> np.ndarray:
    """
    Computes KPIs tensor with the respective weights size and output memory size for each activation configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize)
            (not used in this method).
        fw_impl: FrameworkImplementation object with specific framework methods implementation(not used in this method).

    Returns: A 2D tensor of nodes' weights memory sizes and activation output memory size.
    Note that the vector is not necessarily of the same length as the given config.

    """
    weights_activation_memory = []

    # Go over all nodes that should be taken into consideration when computing the weights or
    # activation KPI (all configurable nodes).
    for node_idx, n in enumerate(graph.get_configurable_sorted_nodes()):
        node_qc = n.candidates_quantization_cfg[mp_cfg[node_idx]]
        node_weights_nbits = node_qc.weights_quantization_cfg.weights_n_bits
        node_activation_nbits = node_qc.activation_quantization_cfg.activation_n_bits

        # Compute node's weights memory (if no weights to quantize then set to 0)
        node_weights_memory_in_bytes = 0
        origin_node = n
        if n.is_weights_quantization_enabled() and not n.is_all_weights_candidates_equal():
            n = _get_origin_weights_node(n)
            node_num_weights_params = 0
            for attr in fw_info.get_kernel_op_attributes(n.type):
                if attr is not None:
                    node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]
            node_weights_memory_in_bytes = node_num_weights_params * node_weights_nbits / BITS_TO_BYTES

        n = origin_node

        # Compute node's activation memory (if node's activation are not being quantized then set to 0)
        node_activation_memory_in_bytes = 0
        if n.is_activation_quantization_enabled() and not n.is_all_activation_candidates_equal():
            n = _get_origin_activation_node(n)
            node_output_size = n.get_total_output_params()
            node_activation_memory_in_bytes = node_output_size * node_activation_nbits / BITS_TO_BYTES

        weights_activation_memory.append(np.array([node_weights_memory_in_bytes, node_activation_memory_in_bytes]))

    return np.array(weights_activation_memory)


def bops_kpi(mp_cfg: List[int],
             graph: Graph,
             fw_info: FrameworkInfo,
             fw_impl: FrameworkImplementation,
             set_constraints: bool = True) -> np.ndarray:
    """
    Computes a KPIs vector with the respective bit-operations (BOPS) count for each configurable node,
    according to the given mixed-precision configuration of a virtual graph with composed nodes.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with specific framework methods implementation.
        set_constraints: A flag for utilizing the method for KPI computation of a
            given config not for LP formalization purposes.

    Returns: A vector of node's BOPS count.
    Note that the vector is not necessarily of the same length as the given config.

    """

    if not set_constraints:
        return _bops_kpi(mp_cfg,
                         graph,
                         fw_info,
                         fw_impl)

    virtual_bops_nodes = [n for n in graph.get_topo_sorted_nodes() if isinstance(n, VirtualActivationWeightsNode)]

    mp_nodes = graph.get_configurable_sorted_nodes_names()
    bops = [n.get_bops_count(fw_impl, fw_info, candidate_idx=_get_node_cfg_idx(n, mp_cfg, mp_nodes)) for n in virtual_bops_nodes]

    return np.array(bops)


def _bops_kpi(mp_cfg: List[int],
              graph: Graph,
              fw_info: FrameworkInfo,
              fw_impl: FrameworkImplementation) -> np.ndarray:
    """
    Computes a KPIs vector with the respective bit-operations (BOPS) count for each configurable node,
    according to the given mixed-precision configuration of an original graph.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with specific framework methods implementation.

    Returns: A vector of node's BOPS count.

    """

    mp_nodes = graph.get_configurable_sorted_nodes_names()

    # Go over all nodes that should be taken into consideration when computing the BOPS KPI.
    bops = []
    for n in graph.get_topo_sorted_nodes():
        if n.has_weights_to_quantize(fw_info):
            # If node doesn't have weights then its MAC count is 0, and we shouldn't consider it in the BOPS count.
            incoming_edges = graph.incoming_edges(n, sort_by_attr=EDGE_SINK_INDEX)
            if len(incoming_edges) != 1:
                Logger.critical(f"Can't compute BOPS metric for node {n.name} with multiple inputs.")

            input_activation_node = incoming_edges[0].source_node
            if len(graph.out_edges(input_activation_node)) > 1:
                # In the case where the activation node has multiple outgoing edges
                # we don't consider this edge in the BOPS KPI calculation
                continue

            input_activation_node_cfg = input_activation_node.candidates_quantization_cfg[_get_node_cfg_idx(input_activation_node, mp_cfg, mp_nodes)]

            node_mac = fw_impl.get_node_mac_operations(n, fw_info)

            node_qc = n.candidates_quantization_cfg[_get_node_cfg_idx(n, mp_cfg, mp_nodes)]
            node_weights_nbits = node_qc.weights_quantization_cfg.weights_n_bits if \
                node_qc.weights_quantization_cfg.enable_weights_quantization else FLOAT_BITWIDTH
            input_activation_nbits = input_activation_node_cfg.activation_quantization_cfg.activation_n_bits if \
                input_activation_node_cfg.activation_quantization_cfg.enable_activation_quantization else FLOAT_BITWIDTH

            node_bops = node_weights_nbits * input_activation_nbits * node_mac
            bops.append(node_bops)

    return np.array(bops)


def _get_node_cfg_idx(node: BaseNode, mp_cfg: List[int], sorted_configurable_nodes_names: List[str]) -> int:
    """
    Returns the index of a node's quantization configuration candidate according to the given
    mixed-precision configuration. If the node is not configurable, then it must have a single configuration,
    therefore, the index 0 is returned.

    Args:
        node: A node to get its candidate configuration index.
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        sorted_configurable_nodes_names: A list of configurable nodes names.

    Returns: An index (integer) of a node's quantization configuration candidate.
    """

    if node.name in sorted_configurable_nodes_names:
        node_idx = sorted_configurable_nodes_names.index(node.name)
        return mp_cfg[node_idx]
    else:
        assert len(node.candidates_quantization_cfg) > 0, \
            "Any node should have at least one candidate configuration."
        return 0


def _get_origin_weights_node(n: BaseNode) -> BaseNode:
    """
    In case we run a KPI computation on a virtual graph,
    this method is used to retrieve the original node out of a virtual weights node,

    Args:
        n: A possibly virtual node.

    Returns: A node from the original (non-virtual) graph which the given node represents.

    """

    if isinstance(n, VirtualActivationWeightsNode):
        return n.original_weights_node
    if isinstance(n, VirtualSplitWeightsNode):
        return n.origin_node

    return n


def _get_origin_activation_node(n: BaseNode) -> BaseNode:
    """
    In case we run a KPI computation on a virtual graph,
    this method is used to retrieve the original node out of a virtual activation node,

    Args:
        n: A possibly virtual node.

    Returns: A node from the original (non-virtual) graph which the given node represents.

    """

    if isinstance(n, VirtualActivationWeightsNode):
        return n.original_activation_node
    if isinstance(n, VirtualSplitActivationNode):
        return n.origin_node

    return n


class MpKpiMetric(Enum):
    """
    Defines kpi computation functions that can be used to compute KPI for a given target for a given mp config.
    The enum values can be used to call a function on a set of arguments.

     WEIGHTS_SIZE - applies the weights_size_kpi function

     ACTIVATION_OUTPUT_SIZE - applies the activation_output_size_kpi function

     TOTAL_WEIGHTS_ACTIVATION_SIZE - applies the total_weights_activation_kpi function

     BOPS_COUNT - applies the bops_kpi function

    """

    WEIGHTS_SIZE = partial(weights_size_kpi)
    ACTIVATION_OUTPUT_SIZE = partial(activation_output_size_kpi)
    TOTAL_WEIGHTS_ACTIVATION_SIZE = partial(total_weights_activation_kpi)
    BOPS_COUNT = partial(bops_kpi)

    def __call__(self, *args):
        return self.value(*args)
