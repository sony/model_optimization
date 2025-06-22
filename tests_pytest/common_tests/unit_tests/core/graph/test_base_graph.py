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
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness
from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode, NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import power_of_two_selection_histogram

def build_mock_fusing_info(nodes, idx):
    """
    Creates a mock FusingInfo object that simulates the behavior of fusing information in a graph.
    """

    OpQCfg = Mock(spec=NodeActivationQuantizationConfig)
    OpQCfg.activation_n_bits = 16
    OpQCfg.signedness = Signedness.AUTO
    OpQCfg.activation_quantization_method = QuantizationMethod.POWER_OF_TWO
    OpQCfg.activation_quantization_params_fn = power_of_two_selection_histogram

    fusing_info = Mock(spec=FusingInfo)
    fusing_info.get_inner_fln_nodes.return_value = [nodes[0], nodes[1]]
    
    if idx == 1:
        OpQCfg.enable_activation_quantization = True
        fusing_info.get_fused_op_quantization_config.side_effect = [OpQCfg, None]
    elif idx == 2:
        fusing_info.get_fused_op_quantization_config.side_effect = [None, None]
    else:
        OpQCfg.enable_activation_quantization = False
        fusing_info.get_fused_op_quantization_config.side_effect = [OpQCfg, None]

    return fusing_info

def build_mock_node(name, layer_class):
    """
    Creates mock nodes representing a simple neural network structure.
    """
    node = Mock(spec=BaseNode)
    node.name = name
    node.layer_class = layer_class

    activation_quantization_cfg = Mock(spec=NodeActivationQuantizationConfig)
    activation_quantization_cfg.quant_mode = Mock()
    candidate_quantization_config = Mock(spec=CandidateNodeQuantizationConfig)
    candidate_quantization_config.activation_quantization_cfg = activation_quantization_cfg

    node.candidates_quantization_cfg = [candidate_quantization_config]

    return node


class TestGraph:
    
    @pytest.mark.parametrize(("idx"), [
        0,
        1,
        2,
    ])
    def test_override_fused_node_activation_quantization_candidates(self, idx):
        """
        Test the override_fused_node_activation_quantization_candidates function for a graph with multiple nodes and configurations.
        """
        ### Create Test Nodes
        mock_nodes = []
        mock_nodes.append(build_mock_node(name='conv', layer_class='Conv2d'))
        mock_nodes.append(build_mock_node(name='fc', layer_class='Linear'))

        ### Create a mock graph
        ### Note: Generate the graph first because fusing_info cannot be set without it.
        ###       In the following Mock, use wraps to mock everything except fusing_info.
        real_graph = Graph("dummy", [], [], [], [])
        real_graph.fusing_info = build_mock_fusing_info(mock_nodes, idx) 
        
        graph = Mock(spec=Graph, wraps=real_graph)
        graph.nodes = mock_nodes

        ### call override_fused_node_activation_quantization_candidates
        graph.override_fused_node_activation_quantization_candidates()

        ### Check if the ActivationQuantization settings set on the graph nodes match the expected values
        nodes = list(graph.nodes)
        
        if idx == 1:
            ### Check if the first node ActivationQuantization settings match the expected values
            assert nodes[0].candidates_quantization_cfg[0].activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_QUANT
            assert nodes[0].candidates_quantization_cfg[0].activation_quantization_cfg.activation_n_bits == 16
            assert nodes[0].candidates_quantization_cfg[0].activation_quantization_cfg.signedness == Signedness.AUTO
            assert nodes[0].candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_method == QuantizationMethod.POWER_OF_TWO
            assert nodes[0].candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_params_fn == power_of_two_selection_histogram
            ### Check if the second node ActivationQuantization settings match the expected values
            assert nodes[1].candidates_quantization_cfg[0].activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_NO_QUANT
        else:
            ### Check if the first node ActivationQuantization settings match the expected values
            assert nodes[0].candidates_quantization_cfg[0].activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_NO_QUANT
