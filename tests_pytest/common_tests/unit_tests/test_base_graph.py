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
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.constants import FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG
from model_compression_toolkit.core.common.quantization.quantization_params_generation.lut_kmeans_params import lut_kmeans_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import symmetric_selection_histogram
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness
from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc
from tests_pytest._test_util.graph_builder_utils import build_nbits_qc as build_qc
from tests_pytest.common_tests.unit_tests.test_filter_nodes_candidates import create_mock_base_node
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode
import copy

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


class TestGraph:
    @pytest.fixture(autouse=True)
    def setup_mock_graph_and_nodes(self):
        """
        Set up mock objects for testing the disable_fused_nodes_activation_quantization modules.

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
                                                    candidates_quantization_cfg=copy.deepcopy(candidate_single)))
        mock_nodes_list.append(create_mock_base_node(name='relu_1', layer_class='ReLU', 
                                                    is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidate_single)))
        mock_nodes_list.append(create_mock_base_node(name='conv_2', layer_class='Conv2d', 
                                                    is_weights_quantization_enabled=True, is_fln_quantization=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidate_single)))
        mock_nodes_list.append(create_mock_base_node(name='tanh', layer_class='Tanh', 
                                                    is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidate_single)))
        mock_nodes_list.append(create_mock_base_node(name='conv_3', layer_class='Conv2d', 
                                                    is_weights_quantization_enabled=True, is_fln_quantization=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidates_multiple)))
        mock_nodes_list.append(create_mock_base_node(name='relu_2', layer_class='ReLU', 
                                                    is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidates_multiple)))
        mock_nodes_list.append(create_mock_base_node(name='flatten', layer_class='Flatten', 
                                                    is_quantization_preserving=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidate_single)))
        mock_nodes_list.append(create_mock_base_node(name='linear', layer_class='Linear', 
                                                    is_weights_quantization_enabled=True, is_activation_quantization_enabled=True,
                                                    candidates_quantization_cfg=copy.deepcopy(candidates_multiple)))
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
        Set up and create expected values and candidates for tests related to the disable_fused_nodes_activation_quantization modules.
        """
        def create_exp_candidate_cfg(candidate, quant_mode, n_bits, actq_params_fn, actq_method, signedness=None):
            ret_candidate = deepcopy(candidate)
            ret_c_actq_cfg = ret_candidate.activation_quantization_cfg

            ret_c_actq_cfg.quant_mode = quant_mode    
            ret_c_actq_cfg.activation_n_bits = n_bits
            ret_c_actq_cfg.activation_quantization_fn = actq_params_fn   ### same as the activation_quantization_params_fn
            ret_c_actq_cfg.activation_quantization_method = actq_method
            ret_c_actq_cfg.activation_quantization_params_fn = actq_params_fn
            if signedness is not None:
                ret_c_actq_cfg.signedness = signedness
            
            return ret_candidate

        ### expected is test_disable_fused_nodes_activation_quantization 
        exp_candidate_base1 = build_qc(a_nbits=8, a_enable=True, w_attr={'weight': (8, True)},
                                        activation_quantization_fn=symmetric_selection_histogram,
                                        activation_quantization_method=QuantizationMethod.SYMMETRIC)
        exp_candidate_base2 = build_qc(a_nbits=4, a_enable=True, w_attr={'weight': (4, True)},
                                        activation_quantization_fn=symmetric_selection_histogram,
                                        activation_quantization_method=QuantizationMethod.SYMMETRIC)
        
        exp_actq_cfg_params_dict1 = {'quant_mode': ActivationQuantizationMode.FLN_QUANT,
                                     'n_bits': 4,
                                     'actq_params_fn': lut_kmeans_histogram,
                                     'actq_method': QuantizationMethod.LUT_POT_QUANTIZER,
                                     'signedness': Signedness.AUTO}
        
        exp_actq_cfg_params_dict2 = {'quant_mode':ActivationQuantizationMode.FLN_NO_QUANT,
                                     'n_bits': 8,
                                     'actq_params_fn': symmetric_selection_histogram,
                                     'actq_method': QuantizationMethod.SYMMETRIC}

        exp_actq_cfg_params_dict3 = {'quant_mode':ActivationQuantizationMode.QUANT,
                                     'n_bits': 8,
                                     'actq_params_fn': symmetric_selection_histogram,
                                     'actq_method': QuantizationMethod.SYMMETRIC}

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
        flatten_qc_cfg = [create_exp_candidate_cfg(exp_candidate_base1, **exp_actq_cfg_params_dict3)]
        ### Expected values when unchanged
        linear_qc_cfg = [exp_candidate_base1, exp_candidate_base2]

        self.exp_node_config_list = [conv_1_qc_cfg, relu_1_qc_cfg,
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
            assert test_actq_cfg.quant_mode == exp_actq_cfg.quant_mode
            assert test_actq_cfg.activation_n_bits == exp_actq_cfg.activation_n_bits
            assert test_actq_cfg.activation_quantization_method == exp_actq_cfg.activation_quantization_method
            assert test_actq_cfg.activation_quantization_params_fn == exp_actq_cfg.activation_quantization_params_fn
            assert test_actq_cfg.signedness == exp_actq_cfg.signedness

    def test_disable_fused_nodes_activation_quantization(self, fusing_info_generator_with_qconfig):
        """
        Test the disable_fused_nodes_activation_quantizatio function for a graph with multiple nodes and configurations.
        """
        graph = self.graph
        fw_info = self.fw_info
        fw_info.get_kernel_op_attributes.return_value = ['weight']
        
        graph.fw_info = fw_info
        graph.fusing_info = fusing_info_generator_with_qconfig.generate_fusing_info(graph)

        # call function disable_fused_nodes_activation_quantization
        graph.disable_fused_nodes_activation_quantization()

        ### Check if the ActivationQuantization settings set on the graph nodes match the expected values
        for node, exp_qc in zip(list(graph.nodes), self.exp_node_config_list):
            self.check_candidates_activation_qcfg(node.candidates_quantization_cfg, exp_qc)
