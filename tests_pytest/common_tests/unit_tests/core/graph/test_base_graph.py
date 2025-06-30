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
import itertools

import pytest
from unittest.mock import Mock

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode, NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig, NodeQuantizationConfig
from tests_pytest._test_util.graph_builder_utils import build_node


def build_mock_fusing_info(nodes, idx):
    """
    Creates a mock FusingInfo object that simulates the behavior of fusing information in a graph.
    """

    OpQCfg = Mock(spec=NodeActivationQuantizationConfig)
    OpQCfg.activation_n_bits = 16
    OpQCfg.signedness = Signedness.AUTO
    OpQCfg.activation_quantization_method = QuantizationMethod.POWER_OF_TWO
    OpQCfg.quantization_preserving = False

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

def build_mock_node(name, layer_class, w_cfgs):
    """
    Creates mock nodes representing a simple neural network structure.
    """
    node = build_node(name, layer_class=layer_class)

    def eq(self_, other):
        return self_.activation_n_bits == other.activation_n_bits and self_._quant_mode == other.quant_mode
    a_cfgs = [Mock(spec=NodeActivationQuantizationConfig,
                   quant_mode=Mock(),
                   activation_n_bits=b,
                   __eq__=eq) for b in [5, 6]]

    qcs = [CandidateNodeQuantizationConfig(a_cfg, w_cfg) for a_cfg, w_cfg in itertools.product(a_cfgs, w_cfgs)]

    node.quantization_cfg = NodeQuantizationConfig(base_quantization_cfg=qcs[0],
                                                   candidates_quantization_cfg=qcs,
                                                   validate=False)
    return node


class TestGraph:
    
    @pytest.mark.parametrize(("idx"), [
        1,
        2,
        3,
    ])
    def test_override_fused_node_activation_quantization_candidates(self, idx, patch_fw_info):
        """
        Test the override_fused_node_activation_quantization_candidates function for a graph with multiple nodes and configurations.
        """
        ### Create Test Nodes
        mock_nodes = []
        w_cfgs = [Mock(), Mock()]
        mock_nodes.append(build_mock_node(name='conv', layer_class='Conv2d', w_cfgs=w_cfgs))
        mock_nodes.append(build_mock_node(name='fc', layer_class='Linear', w_cfgs=w_cfgs[:1]))

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
            # Check if the first node ActivationQuantization settings match the expected values
            # Weight mp configs are preserved, all candidates have the new activation config and duplicates are removed
            qcs0 = nodes[0].quantization_cfg.candidates_quantization_cfg
            assert len(qcs0) == 2
            for i, qc in enumerate(qcs0):
                assert qc.activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_QUANT
                assert qc.activation_quantization_cfg.activation_n_bits == 16
                assert qc.activation_quantization_cfg.signedness == Signedness.AUTO
                assert qc.activation_quantization_cfg.activation_quantization_method == QuantizationMethod.POWER_OF_TWO
                assert qc.weights_quantization_cfg == w_cfgs[i]
            base_cfg0 = nodes[0].quantization_cfg.base_quantization_cfg
            assert base_cfg0.activation_quantization_cfg.activation_n_bits == 16
            assert base_cfg0.activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_QUANT
            ### Check if the second node ActivationQuantization settings match the expected values
            # activations are fln-disabled, duplicates are removed even though orig activation configs differ in nbits
            qcs1 = nodes[1].quantization_cfg.candidates_quantization_cfg
            assert len(qcs1) == 1
            assert qcs1[0].activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_NO_QUANT
            assert qcs1[0].weights_quantization_cfg == w_cfgs[0]
            assert (nodes[1].quantization_cfg.base_quantization_cfg.
                    activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_NO_QUANT)

        else:
            ### Check if the first node ActivationQuantization settings match the expected values
            qcs0 = nodes[0].quantization_cfg.candidates_quantization_cfg
            assert len(qcs0) == 2
            for i, qc in enumerate(qcs0):
                assert qc.activation_quantization_cfg.quant_mode == ActivationQuantizationMode.FLN_NO_QUANT
                assert qc.weights_quantization_cfg == w_cfgs[i]
