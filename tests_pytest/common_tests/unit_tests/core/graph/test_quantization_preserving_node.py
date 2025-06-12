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
import pytest

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.framework_info import set_fw_info

from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc


class TestQuantizationPreservingNode:
    @pytest.fixture(autouse=True)
    def setup(self, fw_info_mock):
        set_fw_info(fw_info_mock)

    def test_activation_preserving_candidate(self):
        """ Tests that the correct activation quantization candidate is selected. """
        n1 = build_node('qact_node', qcs=[build_nbits_qc()])
        n2 = build_node('qp1a_node', qcs=[build_nbits_qc(a_enable=False, q_preserving=True)])
        n3 = build_node('qp1b_node', qcs=[build_nbits_qc(a_enable=False, q_preserving=True)])
        n4 = build_node('qp2a_node', qcs=[build_nbits_qc()])
        n5 = build_node('qp2b_node', qcs=[build_nbits_qc(a_enable=False, q_preserving=True)])
        graph = Graph('g', input_nodes=[n1], nodes=[n2, n4], output_nodes=[n3, n5],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0),
                                 Edge(n1, n4, 0, 0), Edge(n4, n5, 0, 0)])

        assert graph.retrieve_preserved_quantization_node(n2) is n1
        assert graph.retrieve_preserved_quantization_node(n3) is n1
        assert graph.retrieve_preserved_quantization_node(n4) is n4
        assert graph.retrieve_preserved_quantization_node(n5) is n4

    def test_activation_preserving_disable_for_multi_input_node(self):
        """ Tests that the retrieve_preserved_quantization_node raises an assertion error if node has more than 1 input. """
        n1 = build_node('qact_node', qcs=[build_nbits_qc()])
        n2 = build_node('qp1a_node', qcs=[build_nbits_qc(a_enable=False, q_preserving=True)])
        n3 = build_node('qact1b_node', qcs=[build_nbits_qc()])
        n4 = build_node('qp2_node', qcs=[build_nbits_qc(a_enable=False, q_preserving=True)])
        graph = Graph('g', input_nodes=[n1], nodes=[n2, n3], output_nodes=[n4],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n1, n3, 0, 0),
                                 Edge(n2, n4, 0, 0), Edge(n2, n4, 0, 0)])

        with pytest.raises(AssertionError, match="Activation preserving node should have only 1 input"):
            graph.retrieve_preserved_quantization_node(n4)
