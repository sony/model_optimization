# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from copy import deepcopy 

import pytest
from unittest.mock import Mock

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode, NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_node_candidates
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from mct_quantizers import QuantizationMethod

def build_mock_node(name, layer_class, wqe_flag):
    """
    Creates mock nodes representing a simple neural network structure.
    """
    node = Mock(spec=BaseNode)
    node.name = name
    node.layer_class = layer_class

    node.is_no_quantization.return_value = True
    node.is_fln_no_quantization.return_value = False
    node.is_weights_quantization_enabled.return_value = wqe_flag

    activation_quantization_cfg = Mock(spec=NodeActivationQuantizationConfig)
    activation_quantization_cfg.quant_mode = Mock()
    candidate_quantization_config = Mock(spec=CandidateNodeQuantizationConfig)
    candidate_quantization_config.activation_quantization_cfg = activation_quantization_cfg
    candidate_quantization_config.weights_quantization_cfg = Mock()

    node.candidates_quantization_cfg = [candidate_quantization_config]

    return node

@pytest.mark.parametrize(("wqe_flag"), [
    False,
    True,   ### QUANT layer : ReLU
])
def test_filter_node_candidates(wqe_flag):
    """
    Test the filter_node_candidates function for a graph with multiple nodes and configurations.
    """
    ### Create Test Nodes
    mock_nodes = []
    mock_nodes.append(build_mock_node(name='conv', layer_class='Conv2d', wqe_flag=wqe_flag))
    ### Create a mock graph
    ### Note: Generate the graph first because fusing_info cannot be set without it.
    ###       In the following Mock, use wraps to mock everything except fusing_info.
    real_graph = Graph("dummy", [], [], [], [])
    
    graph = Mock(spec=Graph, wraps=real_graph)
    graph.nodes = mock_nodes
    ### call override_fused_node_activation_quantization_candidates
    graph.override_fused_node_activation_quantization_candidates()
    fw_info = Mock()
    fw_info.get_kernel_op_attributes.return_value = ['weight']
    output_candidates = filter_node_candidates(graph.nodes[0], fw_info)
    assert output_candidates[0].activation_quantization_cfg.activation_n_bits == FLOAT_BITWIDTH
    assert output_candidates[0].activation_quantization_cfg.activation_quantization_method == QuantizationMethod.POWER_OF_TWO
