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

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import QuantizationConfig

from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness

from tests_pytest._test_util.graph_builder_utils import build_node


def build_quant_preserving_qc(a_enable=True, qp_enable=False) -> CandidateNodeQuantizationConfig:
    """
    Build quantization config with enabling/disabling quantization and quantization preserving only.

    Args:
        a_enable: whether to enable activation quantization.
        qp_enable: whether to enable activation quantization preserving flag.

    Returns:
        CandidateNodeQuantizationConfig object.

    """
    qc = QuantizationConfig()
    # positional attrs are set via default weight config (so all pos attrs have the same q config)
    op_cfg = OpQuantizationConfig(
        # canonical names (as 'kernel')
        attr_weights_configs_mapping={},
        activation_n_bits=8,
        enable_activation_quantization=a_enable,
        default_weight_attr_config=AttributeQuantizationConfig(weights_n_bits=8,
                                                               enable_weights_quantization=False),
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        quantization_preserving=qp_enable,
        supported_input_activation_n_bits=[2, 4, 8],
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=None,
        signedness=Signedness.AUTO
    )
    a_qcfg = NodeActivationQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                              activation_quantization_fn=None,
                                              activation_quantization_params_fn=None)
    # full names from the layers
    w_qcfg = NodeWeightsQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                           weights_channels_axis=None,
                                           node_attrs_list=[])
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg,
                                         weights_quantization_cfg=w_qcfg)

    return qc


class TestQuantizationPreservingNode:

    def test_activation_preserving_candidate(self):
        """ Tests that the correct activation quantization candidate is selected. """
        n1 = build_node('qact_node', qcs=[build_quant_preserving_qc()])
        n2 = build_node('qp1a_node', qcs=[build_quant_preserving_qc(a_enable=False, qp_enable=True)])
        n3 = build_node('qp1b_node', qcs=[build_quant_preserving_qc(a_enable=False, qp_enable=True)])
        n4 = build_node('qp2a_node', qcs=[build_quant_preserving_qc()])
        n5 = build_node('qp2b_node', qcs=[build_quant_preserving_qc(a_enable=False, qp_enable=True)])
        graph = Graph('g', input_nodes=[n1], nodes=[n2, n4], output_nodes=[n3, n5],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0),
                                 Edge(n1, n4, 0, 0), Edge(n4, n5, 0, 0)])

        assert graph.retrieve_preserved_quantization_node(n2) is n1
        assert graph.retrieve_preserved_quantization_node(n3) is n1
        assert graph.retrieve_preserved_quantization_node(n4) is n4
        assert graph.retrieve_preserved_quantization_node(n5) is n4
