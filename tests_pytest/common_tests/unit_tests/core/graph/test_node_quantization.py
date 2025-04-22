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
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge

from unittest.mock import Mock
from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc, DummyLayer
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import set_quantization_configs_to_node
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import QuantizationConfigOptions, \
    OpQuantizationConfig, AttributeQuantizationConfig, Signedness
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO



from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities


class TestSetNodeQuantizationConfig:

    @staticmethod
    def _get_op_config():
        aqc = AttributeQuantizationConfig()
        return OpQuantizationConfig(default_weight_attr_config=aqc,
                                    attr_weights_configs_mapping={'w': aqc},
                                    activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                    activation_n_bits=7,
                                    supported_input_activation_n_bits=7,
                                    enable_activation_quantization=False,
                                    quantization_preserving=True,
                                    signedness=Signedness.AUTO)

    def test_activation_preserving_with_2_inputs(self, fw_impl_mock):
        """ Tests that . """
        n1 = build_node('in1_node')
        n2 = build_node('in2_node')
        n3 = build_node('qp_node')
        n4 = build_node('qp2_node')
        graph = Graph('g', input_nodes=[n1, n2], nodes=[n3], output_nodes=[n4],
                      edge_list=[Edge(n1, n3, 0, 0), Edge(n2, n3, 0, 0),
                                 Edge(n3, n4, 0, 0)])

        fqc = Mock(filterlayer2qco={DummyLayer: QuantizationConfigOptions(quantization_configurations=[self._get_op_config()])},
                   layer2qco={DummyLayer: QuantizationConfigOptions(quantization_configurations=[self._get_op_config()])})
        set_quantization_configs_to_node(n3, graph, QuantizationConfig(), DEFAULT_PYTORCH_INFO, fqc)
        set_quantization_configs_to_node(n4, graph, QuantizationConfig(), DEFAULT_PYTORCH_INFO, fqc)
        assert not n3.is_quantization_preserving() and not n3.is_activation_quantization_enabled()
        assert not n4.is_quantization_preserving() and not n4.is_activation_quantization_enabled()
