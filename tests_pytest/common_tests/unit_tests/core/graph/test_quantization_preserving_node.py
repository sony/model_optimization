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
import numpy as np
import pytest

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge

from tests_pytest._test_util.graph_builder_utils import build_node, DummyLayer
from tests_pytest._test_util.graph_builder_utils import build_nbits_qc as build_qc


class DummyLayerWKernel:
    pass


class TestQuantizationPreservingNode:

    def test_activation_preserving_candidate(self):
        """ Tests that the correct activation quantization candidate is selected. """
        n1 = build_node('qact_node', qcs=[build_qc()])
        n2 = build_node('qp1a_node', qcs=[build_qc(a_enable=False, qp_enable=True)])
        n3 = build_node('qp1b_node', qcs=[build_qc(a_enable=False, qp_enable=True)])
        n4 = build_node('qp2a_node', qcs=[build_qc()])
        n5 = build_node('qp2b_node', qcs=[build_qc(a_enable=False, qp_enable=True)])
        graph = Graph('g', input_nodes=[n1], nodes=[n2, n4], output_nodes=[n3, n5],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0),
                                 Edge(n1, n4, 0, 0), Edge(n4, n5, 0, 0)])

        assert graph.get_act_config_node(n2) is n1
        assert graph.get_act_config_node(n3) is n1
        assert graph.get_act_config_node(n4) is n4
        assert graph.get_act_config_node(n5) is n4
