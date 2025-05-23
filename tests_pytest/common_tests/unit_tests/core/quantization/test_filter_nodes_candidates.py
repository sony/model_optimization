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
from unittest.mock import Mock, MagicMock

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.core.common import DEFAULTCONFIG, Graph, BaseNode
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates, filter_node_candidates
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig
from model_compression_toolkit.constants import FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG, FLOAT_BITWIDTH
from model_compression_toolkit.core.common.quantization.quantization_params_generation.lut_kmeans_params import lut_kmeans_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import power_of_two_selection_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import symmetric_selection_histogram
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness

from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc
from tests_pytest._test_util.graph_builder_utils import build_nbits_qc as build_qc

# Setup TEST_QC and TEST_QCO for testing.
TEST_QC = generate_test_op_qc(**generate_test_attr_configs(),
                              activation_n_bits=4,
                              activation_quantization_method=QuantizationMethod.LUT_POT_QUANTIZER)


@pytest.fixture
def fusing_patterns_with_qconfig():
    """
    - Returns predefined fusing patterns: Conv2D + ReLU and  Conv2D + Tanh.
    """
    return [{FUSED_LAYER_PATTERN: ["Conv2d", "ReLU"], FUSED_OP_QUANT_CONFIG: TEST_QC},
            {FUSED_LAYER_PATTERN: ["Conv2d", "Tanh"], FUSED_OP_QUANT_CONFIG: None}]


@pytest.fixture
def fusing_info_generator_with_qconfig(fusing_patterns_with_qconfig):
    """
    Creates a FusingInfoGenerator using the fusing patterns.
    """
    return FusingInfoGenerator(fusing_patterns_with_qconfig)


def create_mock_base_node(name: str, layer_class: str,
                          is_weights_quantization_enabled: bool = False,
                          is_activation_quantization_enabled: bool = False,
                          is_fln_quantization: bool = False,
                          is_quantization_preserving: bool = False,
                          candidates_quantization_cfg: CandidateNodeQuantizationConfig = None):
    """
    Function for creating the mock nodes required for a simple neural network structure.
    """

    dummy_initalize = {'framework_attr': {},
                       'input_shape': (),
                       'output_shape': (),
                       'weights': {}}

    real_node = BaseNode(name=name, layer_class=layer_class, **dummy_initalize)

    node = Mock(spec=real_node)
    node.is_match_type = real_node.is_match_type
    node.layer_class = layer_class
    node.name = name

    node.candidates_quantization_cfg = candidates_quantization_cfg

    node.is_weights_quantization_enabled.return_value = is_weights_quantization_enabled
    node.is_activation_quantization_enabled.return_value = is_activation_quantization_enabled
    node.is_quantization_preserving.return_value = is_quantization_preserving 
    node.is_fln_quantization.return_value = is_fln_quantization

    return node


class TestFilterNodesCandidates:
    @pytest.fixture(autouse=True)
    def setup_mock_graph_and_nodes(self):
        """
        Set up mock objects for testing the filter_nodes_candidates modules.

        Mocks the behavior of nodes in a graph that support activation and weight quantization settings.
        Creates a graph containing nodes with multiple quantization candidates
        to test whether the expected configurations are set correctly.
        """

        self.fw_info = Mock()

        candidate1 = build_qc(a_nbits=8, a_enable=True, w_attr={'weight': (8, True)}, q_preserving=False,
                            activation_quantization_fn=symmetric_selection_histogram,
                            activation_quantization_method=QuantizationMethod.SYMMETRIC)
        candidate2 = build_qc(a_nbits=4, a_enable=True, w_attr={'weight': (4, True)}, q_preserving=False,
                            activation_quantization_fn=symmetric_selection_histogram,
                            activation_quantization_method=QuantizationMethod.SYMMETRIC)
        candidate_single  = [candidate1]
        candidates_multiple = [candidate1, candidate2]

        # Create Test Nodes
        mock_nodes_list = []
        mock_nodes_list.append(create_mock_base_node(name='conv_1', layer_class='Conv2d', 
                                                    is_weights_quantization_enabled=True, is_fln_quantization=True,
                                                    candidates_quantization_cfg=candidate_single))
        mock_nodes_list.append(create_mock_base_node(name='relu_1', layer_class='ReLU', 
                                                    is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=candidate_single))
        mock_nodes_list.append(create_mock_base_node(name='conv_2', layer_class='Conv2d', 
                                                    is_weights_quantization_enabled=True, is_fln_quantization=True,
                                                    candidates_quantization_cfg=candidate_single))
        mock_nodes_list.append(create_mock_base_node(name='tanh', layer_class='Tanh', 
                                                    is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=candidate_single))
        mock_nodes_list.append(create_mock_base_node(name='conv_3', layer_class='Conv2d', 
                                                    is_weights_quantization_enabled=True, is_fln_quantization=True,
                                                    candidates_quantization_cfg=candidates_multiple))
        mock_nodes_list.append(create_mock_base_node(name='relu_2', layer_class='ReLU', 
                                                    is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=candidates_multiple))
        mock_nodes_list.append(create_mock_base_node(name='flatten', layer_class='Flatten', 
                                                    is_quantization_preserving=True,
                                                    candidates_quantization_cfg=candidate_single))
        mock_nodes_list.append(create_mock_base_node(name='linear', layer_class='Linear', 
                                                    is_weights_quantization_enabled=True, is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=candidates_multiple))
        self.mock_node = mock_nodes_list

        edges_list = []
        for i in range(1, len(mock_nodes_list), 1):
            edges_list.append(Edge(mock_nodes_list[i-1], mock_nodes_list[i], 0, 0))

        # Create a mock graph
        self.graph = Graph('g',
                            input_nodes=[mock_nodes_list[0]],
                            nodes=mock_nodes_list,
                            output_nodes=[OutTensor(mock_nodes_list[5], 0)],
                            edge_list=edges_list)

    @pytest.fixture(autouse=True)
    def expected_candidates_fixture(self):
        """
        Set up and create expected values and candidates for tests related to the filter_nodes_candidates modules.
        """
        def create_exp_candidate_cfg(candidate, n_bits, actq_params_fn, actq_method, signedness=None):
            ret_candidate = deepcopy(candidate)
            ret_c_actq_cfg = ret_candidate.activation_quantization_cfg

            ret_c_actq_cfg.activation_n_bits = n_bits
            ret_c_actq_cfg.activation_quantization_fn = actq_params_fn   ### same as the activation_quantization_params_fn
            ret_c_actq_cfg.activation_quantization_method = actq_method
            ret_c_actq_cfg.activation_quantization_params_fn = actq_params_fn
            if signedness is not None:
                ret_c_actq_cfg.signedness = signedness
            
            return ret_candidate

        ### expected is test_filter_nodes_candidates 
        exp_candidate_base1 = build_qc(a_nbits=8, a_enable=True, w_attr={'weight': (8, True)},
                                        activation_quantization_fn=symmetric_selection_histogram,
                                        activation_quantization_method=QuantizationMethod.SYMMETRIC)
        exp_candidate_base2 = build_qc(a_nbits=4, a_enable=True, w_attr={'weight': (4, True)},
                                        activation_quantization_fn=symmetric_selection_histogram,
                                        activation_quantization_method=QuantizationMethod.SYMMETRIC)
        
        exp_actq_cfg_params_dict1 = {'n_bits': 4,
                                     'actq_params_fn': lut_kmeans_histogram,
                                     'actq_method': QuantizationMethod.LUT_POT_QUANTIZER,
                                     'signedness': Signedness.AUTO}
        exp_actq_cfg_params_dict2 = {'n_bits': FLOAT_BITWIDTH,
                                     'actq_params_fn': power_of_two_selection_histogram,
                                     'actq_method': QuantizationMethod.POWER_OF_TWO}

        ### Expected candidates when transformed by the qcfg of FusingInfo
        conv_1_qc_cfg = [create_exp_candidate_cfg(exp_candidate_base1, **exp_actq_cfg_params_dict1)]
        ### Expected values when unchanged
        relu_1_qc_cfg = [exp_candidate_base1]
        ### Expected candidates when transformed by FusingInfo where qcfg is None
        conv_2_qc_cfg = [create_exp_candidate_cfg(exp_candidate_base1, **exp_actq_cfg_params_dict2)]
        ### Expected values when unchanged
        tanh_qc_cfg   = [exp_candidate_base1]
        ### Expected candidates with multiple configurations when transformed by the qcfg of FusingInfo
        conv_3_qc_cfg = [create_exp_candidate_cfg(exp_candidate_base1, **exp_actq_cfg_params_dict1),
                         create_exp_candidate_cfg(exp_candidate_base2, **exp_actq_cfg_params_dict1)]
        ### Expected values when unchanged
        relu_2_qc_cfg = [exp_candidate_base1, exp_candidate_base2]
        ### Expected candidates when transformed with preserving set to True
        flatten_qc_cfg = [create_exp_candidate_cfg(exp_candidate_base1, **exp_actq_cfg_params_dict2)]
        ### Expected values when unchanged
        linear_qc_cfg = [exp_candidate_base1, exp_candidate_base2]

        self.exp_filter_nodes_candidates = [conv_1_qc_cfg, relu_1_qc_cfg,
                                            conv_2_qc_cfg, tanh_qc_cfg,
                                            conv_3_qc_cfg, relu_2_qc_cfg,
                                            flatten_qc_cfg, linear_qc_cfg]
    
    def check_candidates_activation_qcfg(self, candidates, exp_candidates):
        """
        Check if the ActivationQuantization settings set on the graph nodes match the expected values
        """
        for test_c, exp_c in zip(candidates, exp_candidates):
            test_actq_cfg = test_c.activation_quantization_cfg
            exp_actq_cfg = exp_c.activation_quantization_cfg

            assert test_actq_cfg.activation_quantization_fn == exp_actq_cfg.activation_quantization_fn
            assert test_actq_cfg.activation_n_bits == exp_actq_cfg.activation_n_bits
            assert test_actq_cfg.activation_quantization_method == exp_actq_cfg.activation_quantization_method
            assert test_actq_cfg.activation_quantization_params_fn == exp_actq_cfg.activation_quantization_params_fn
            assert test_actq_cfg.signedness == exp_actq_cfg.signedness

    @pytest.mark.parametrize(("idx", "op_cfg"), [
        (0, TEST_QC),   ### FLN_QUANT layer : Conv2d
        (1, None),      ### QUANT layer : ReLU
        (2, None),      ### FLN_QUANT layer : Conv2d (with not set QC)
        (4, TEST_QC),   ### FLN_QUANT layer : Conv2d (with multiple QCs)
        (6, None),      ### PRESERVE_QUANT layer : Flatten (with preserving flag set to True)
        (7, None),      ### QUANT layer : Linear (with multiple QCs)
    ])
    def test_filter_node_candidates_updates_activation_quantization_config(self, idx, op_cfg):
        """
        Test the filter_node_candidates function for different node types and configurations.
        """
        nodes = self.mock_node
        fw_info = self.fw_info
        fw_info.get_kernel_op_attributes.return_value = ['weight']

        output_candidates = filter_node_candidates(nodes[idx], fw_info, op_cfg)
        exp_candidates = self.exp_filter_nodes_candidates[idx]

        ### Check if the ActivationQuantization settings match the expected values
        self.check_candidates_activation_qcfg(output_candidates, exp_candidates)

    def test_filter_nodes_candidates_updates_node_activation_quantization_config(self, fusing_info_generator_with_qconfig):
        """
        Test the filter_nodes_candidates function for a graph with multiple nodes and configurations.
        """
        graph = self.graph
        fw_info = self.fw_info
        fw_info.get_kernel_op_attributes.return_value = ['weight']
        
        graph.fw_info = fw_info
        graph.fusing_info = fusing_info_generator_with_qconfig.generate_fusing_info(graph)

        output_graph = filter_nodes_candidates(graph)

        ### Check if the ActivationQuantization settings set on the graph nodes match the expected values
        for node, exp_qc in zip(list(graph.nodes), self.exp_filter_nodes_candidates):
            self.check_candidates_activation_qcfg(node.candidates_quantization_cfg, exp_qc)
