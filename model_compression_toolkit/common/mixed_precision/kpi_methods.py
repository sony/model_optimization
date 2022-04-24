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

import numpy as np


def weights_size_kpi(mp_cfg, graph, fw_info):
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


def activation_output_size_kpi(mp_cfg, graph, fw_info):
    activation_memory = []

    # Go over all nodes that should be taken into consideration when computing the weights KPI.
    mp_nodes = graph.get_configurable_sorted_nodes_names()
    for n in graph.get_sorted_activation_configurable_nodes():
        node_idx = mp_nodes.index(n.name)
        node_qc = n.candidates_quantization_cfg[mp_cfg[node_idx]]
        node_nbits = node_qc.activation_quantization_cfg.activation_n_bits

        node_output_size = n.get_total_output_params()
        node_activation_memory_in_bytes = node_output_size * node_nbits / 8.0
        activation_memory.append(node_activation_memory_in_bytes)

    return np.array(activation_memory)
