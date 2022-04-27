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
from typing import Callable, Any, List, Dict
import numpy as np

from model_compression_toolkit import QuantizationConfig, FrameworkInfo, KPI
from model_compression_toolkit.common import Graph
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.hardware_representation import FrameworkHardwareModel
from model_compression_toolkit.common.post_training_quantization import read_model_to_graph, get_finalized_graph


def compute_kpi_data(in_model: Any,
                     representative_data_gen: Callable,
                     quant_config: QuantizationConfig,
                     fw_hw_model: FrameworkHardwareModel,
                     fw_info: FrameworkInfo,
                     fw_impl: FrameworkImplementation) -> Dict[str, KPI]:
    """
    Compute KPI information that can be relevant for defining target KPI for mixed precision search.
    Calculates minimal and maximal mp configs (based on the provided hw capabilities model) and use them to compute
    maximal activation tensor and sum of weights' parameters (for each of the configurations).

    Args:
        in_model:  Model to build graph from (the model that intended to be quantized).
        representative_data_gen: Dataset used for calibration.
        quant_config: QuantizationConfig containing parameters of how the model should be quantized.
        fw_hw_model: FrameworkHardwareModel object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A dictionary with the results (KPI object) for the maximal and minimal configurations.

    """

    graph = read_model_to_graph(in_model,
                                representative_data_gen,
                                fw_hw_model,
                                fw_info,
                                fw_impl)

    transformed_graph = get_finalized_graph(graph,
                                            quant_config,
                                            fw_info,
                                            tb_w=None,
                                            fw_impl=fw_impl)

    min_cfg = transformed_graph.get_min_candidates_config()
    max_cfg = transformed_graph.get_max_candidates_config()

    ######################################
    # Compute parameters sum
    ######################################
    min_parameters_sum = np.sum(compute_weights_sizes(mp_cfg=min_cfg, graph=graph, fw_info=fw_info))
    max_parameters_sum = np.sum(compute_weights_sizes(mp_cfg=min_cfg, graph=graph, fw_info=fw_info))


    ######################################
    # Compute max activation tensor
    ######################################
    min_precision_largest_tensor = np.max(compute_activation_output_sizes(mp_cfg=min_cfg, graph=graph))
    max_precision_largest_tensor = np.max(compute_activation_output_sizes(mp_cfg=max_cfg, graph=graph))

    return {"min_kpi": KPI(weights_memory=min_parameters_sum, activation_memory=min_precision_largest_tensor),
            "max_kpi": KPI(weights_memory=max_parameters_sum, activation_memory=max_precision_largest_tensor)}


def compute_weights_sizes(mp_cfg: List[int], graph: Graph, fw_info: FrameworkInfo) -> np.ndarray:
    """
    Computes a KPIs vector with the respective weights' memory size for each weigh configurable node,
    according to the given mixed-precision configuration.

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework
            (e.g., attributes of different layers' weights to quantize).

    Returns: A vector of node's weights memory sizes.

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

        node_weights_memory_in_bytes = node_num_weights_params * node_nbits / 8.0
        weights_memory.append(node_weights_memory_in_bytes)

    return np.array(weights_memory)


def compute_activation_output_sizes(mp_cfg: List[int], graph: Graph) -> np.ndarray:
    """
    Computes a KPIs vector with the respective output memory size for each activation configurable node,
    according to the given mixed-precision configuration.
    Note that the configuration includes an index for each configurable node! (not just activation configurable).

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.

    Returns: A vector of node's weights memory sizes.
    Note that the vector is not necessarily of the same length as the given config.

    """

    activation_memory = []

    # Go over all nodes that should be taken into consideration when computing the activation KPI.
    mp_nodes = graph.get_configurable_sorted_nodes_names()
    for n in graph.get_sorted_activation_configurable_nodes():
        node_idx = mp_nodes.index(n.name)
        node_qc = n.candidates_quantization_cfg[mp_cfg[node_idx]]
        node_nbits = node_qc.activation_quantization_cfg.activation_n_bits

        node_output_size = n.get_total_output_params()
        node_activation_memory_in_bytes = node_output_size * node_nbits / 8.0
        activation_memory.append(node_activation_memory_in_bytes)

    return np.array(activation_memory)
