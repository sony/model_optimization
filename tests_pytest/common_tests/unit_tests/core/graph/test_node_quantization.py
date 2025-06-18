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

from unittest.mock import Mock

from model_compression_toolkit.quantization_preparation.load_fqc import set_quantization_configs_to_node
from tests_pytest._test_util.graph_builder_utils import build_node, DummyLayer
from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import QuantizationConfigOptions, \
    OpQuantizationConfig, AttributeQuantizationConfig, Signedness
from mct_quantizers import QuantizationMethod


class PreservingNode:
    pass


class NoActivationQuantNode:
    pass


class TestSetNodeQuantizationConfig:
    @pytest.fixture(autouse=True)
    def setup(self, fw_info_mock):
        set_fw_info(fw_info_mock)

    @staticmethod
    def _get_op_config(activation_n_bits,
                       supported_input_activation_n_bits,
                       enable_activation_quantization,
                       quantization_preserving):
        aqc = AttributeQuantizationConfig()
        return OpQuantizationConfig(default_weight_attr_config=aqc,
                                    attr_weights_configs_mapping={'w': aqc},
                                    activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                    activation_n_bits=activation_n_bits,
                                    supported_input_activation_n_bits=supported_input_activation_n_bits,
                                    enable_activation_quantization=enable_activation_quantization,
                                    quantization_preserving=quantization_preserving,
                                    signedness=Signedness.AUTO)

    def test_activation_preserving_with_2_inputs(self, fw_info_mock):
        """ Tests that . """
        fw_info_mock.activation_quantizer_mapping = {QuantizationMethod.POWER_OF_TWO: lambda x: 0}
        fw_info_mock.get_kernel_op_attribute = lambda x: None

        n1 = build_node('in1_node')
        n2 = build_node('in2_node')
        n3 = build_node('qp_node', layer_class=PreservingNode)
        n4 = build_node('qp2_node', layer_class=PreservingNode)
        qp3 = build_node('qp3_node', layer_class=PreservingNode)
        qp4 = build_node('qp4_node', layer_class=PreservingNode)
        graph = Graph('g', input_nodes=[n1, n2], nodes=[n3, qp3], output_nodes=[n4, qp4],
                      edge_list=[Edge(n1, n3, 0, 0), Edge(n2, n3, 0, 0),
                                 Edge(n3, n4, 0, 0),
                                 Edge(n1, qp3, 0, 0), Edge(qp3, qp4, 0, 0)])
        q_op_config_kwargs = {"activation_n_bits": 7, "supported_input_activation_n_bits": 7,
                            "enable_activation_quantization": True, "quantization_preserving": False}
        qp_op_config_kwargs = {"activation_n_bits": 7, "supported_input_activation_n_bits": 7,
                            "enable_activation_quantization": False, "quantization_preserving": True}
        _filters = {DummyLayer: QuantizationConfigOptions(quantization_configurations=[self._get_op_config(**q_op_config_kwargs)]),
                    PreservingNode: QuantizationConfigOptions(quantization_configurations=[self._get_op_config(**qp_op_config_kwargs)])}
        fqc = Mock(filterlayer2qco=_filters, layer2qco=_filters)

        for n in graph.get_topo_sorted_nodes():
            set_quantization_configs_to_node(n, graph, fqc)
        assert not n3.is_quantization_preserving() and not n3.is_activation_quantization_enabled()
        assert not n4.is_quantization_preserving() and not n4.is_activation_quantization_enabled()
        assert qp3.is_quantization_preserving()
        assert qp4.is_quantization_preserving()

    def test_node_quantization_by_next_nodes(self, fw_info_mock):
        """
        Test that node quantization n_bits is unaffected by preserving next node and not-enabled quantization next node.
        """
        fw_info_mock.activation_quantizer_mapping = {QuantizationMethod.POWER_OF_TWO: lambda x: 0}
        fw_info_mock.get_kernel_op_attribute = lambda x: None

        first_node = build_node('first_node')
        preserving_node = build_node('preserving_node', layer_class=PreservingNode)
        no_quant_node = build_node('no_enabled_quant_node', layer_class=NoActivationQuantNode)
        graph = Graph('g', input_nodes=[first_node], nodes=[preserving_node], output_nodes=[no_quant_node],
                      edge_list=[Edge(first_node, preserving_node, 0, 0),
                                 Edge(preserving_node, no_quant_node, 0, 0)])

        first_node_config_kwargs = {"activation_n_bits": 16,
                                    "supported_input_activation_n_bits": [8, 16],
                                    "enable_activation_quantization": True,
                                    "quantization_preserving": False}

        preserving_node_config_kwargs = {"activation_n_bits": 8,
                                        "supported_input_activation_n_bits": [8, 16],
                                        "enable_activation_quantization": False,
                                        "quantization_preserving": True}

        no_quant_node_config_kwargs = {"activation_n_bits": 8,
                                        "supported_input_activation_n_bits": [8],
                                        "enable_activation_quantization": False,
                                        "quantization_preserving": False}
        _filters = {
            DummyLayer: QuantizationConfigOptions(quantization_configurations=[self._get_op_config(**first_node_config_kwargs)]),
            PreservingNode: QuantizationConfigOptions(quantization_configurations=[self._get_op_config(**preserving_node_config_kwargs)]),
            NoActivationQuantNode: QuantizationConfigOptions(quantization_configurations=[self._get_op_config(**no_quant_node_config_kwargs)]),
        }

        fqc = Mock(filterlayer2qco=_filters, layer2qco=_filters)
        set_quantization_configs_to_node(first_node, graph, fqc)
        set_quantization_configs_to_node(preserving_node, graph, fqc)
        set_quantization_configs_to_node(no_quant_node, graph, fqc)

        assert not first_node.is_quantization_preserving() and first_node.is_activation_quantization_enabled()
        assert preserving_node.is_quantization_preserving() and not preserving_node.is_activation_quantization_enabled()
        assert not no_quant_node.is_quantization_preserving() and not no_quant_node.is_activation_quantization_enabled()

        # assert that first_node n_bits is 16, and isn't affected by its next nodes which supports 8 n_bits only
        assert first_node.candidates_quantization_cfg[0].activation_quantization_cfg.activation_n_bits == 16
