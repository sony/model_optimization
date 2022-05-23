# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit.common import Graph
from model_compression_toolkit.common.constants import BITS_TO_BYTES


def weights_size_kpi(mp_cfg: List[int], graph: Graph, fw_info: FrameworkInfo) -> np.ndarray:
    """
    Computes a KPIs vector with the respective weights' memory size for each weight configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).

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

        node_num_weights_params = 0
        for attr in fw_info.get_kernel_op_attributes(n.type):
            if attr is not None:
                node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]

        node_weights_memory_in_bytes = node_num_weights_params * node_nbits / BITS_TO_BYTES
        weights_memory.append(node_weights_memory_in_bytes)

    return np.array(weights_memory)


def activation_output_size_kpi(mp_cfg: List[int], graph: Graph, fw_info: FrameworkInfo) -> np.ndarray:
    """
    Computes a KPIs vector with the respective output memory size for each activation configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize)
            (not used in this method).

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

        node_output_size = n.get_total_output_params()
        node_activation_memory_in_bytes = node_output_size * node_nbits / BITS_TO_BYTES
        activation_memory.append(node_activation_memory_in_bytes)

    return np.array(activation_memory)


def total_weights_activation_kpi(mp_cfg: List[int], graph: Graph, fw_info: FrameworkInfo) -> np.ndarray:
    """
    Computes KPIs tensor with the respective weights size and output memory size for each activation configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize)
            (not used in this method).

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
        node_num_weights_params = 0
        for attr in fw_info.get_kernel_op_attributes(n.type):
            if attr is not None:
                node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]
        node_weights_memory_in_bytes = node_num_weights_params * node_weights_nbits / BITS_TO_BYTES

        # Compute node's activation memory (if node's activation are not being quantized then set to 0)
        node_activation_memory_in_bytes = 0
        if n.is_activation_quantization_enabled():
            node_output_size = n.get_total_output_params()
            node_activation_memory_in_bytes = node_output_size * node_activation_nbits / BITS_TO_BYTES

        weights_activation_memory.append(np.array([node_weights_memory_in_bytes, node_activation_memory_in_bytes]))

    return np.array(weights_activation_memory)


class MpKpiMetric(Enum):
    """
    Defines kpi computation functions that can be used to compute KPI for a given target for a given mp config.
    The enum values can be used to call a function on a set of arguments.

     WEIGHTS_SIZE - applies the weights_size_kpi function

     ACTIVATION_OUTPUT_SIZE - applies the activation_output_size_kpi function

     TOTAL_WEIGHTS_ACTIVATION_SIZE - applies the total_weights_activation_kpi function

    """

    WEIGHTS_SIZE = partial(weights_size_kpi)
    ACTIVATION_OUTPUT_SIZE = partial(activation_output_size_kpi)
    TOTAL_WEIGHTS_ACTIVATION_SIZE = partial(total_weights_activation_kpi)

    def __call__(self, *args):
        return self.value(*args)
