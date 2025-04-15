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
from typing import List
from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common import BaseNode
from tests_pytest._test_util.graph_builder_utils import DummyLayer
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness
from model_compression_toolkit.core import QuantizationConfig
from mct_quantizers import QuantizationMethod

fw_impl = PytorchImplementation()

class InputLayer:
    pass
class Conv2D:
    pass
class Flatten:
    pass
class Dropout:
    pass
class Linear:
    pass

def build_node(name='node', qcs: List[CandidateNodeQuantizationConfig] = None,
               input_shape=(4, 5, 6), output_shape=(4, 5, 6),
               layer_class=DummyLayer, reuse=False):

    node = BaseNode(name=name,
                    framework_attr={},
                    input_shape=input_shape,
                    output_shape=output_shape,
                    weights={},
                    layer_class=layer_class,
                    reuse=reuse)
    if qcs:
        assert isinstance(qcs, list)
        node.candidates_quantization_cfg = qcs
        node.final_activation_quantization_cfg = node.candidates_quantization_cfg[0].activation_quantization_cfg
    return node

def build_qc(a_nbits=8, a_enable=True, q_preserving=False, aq_params={}):
    qc = QuantizationConfig()
    # positional attrs are set via default weight config (so all pos attrs have the same q config)
    op_cfg = OpQuantizationConfig(
        # canonical names (as 'kernel')
        attr_weights_configs_mapping={},
        activation_n_bits=a_nbits,
        enable_activation_quantization=a_enable,
        default_weight_attr_config=AttributeQuantizationConfig(weights_n_bits=32,
                                                               enable_weights_quantization=False),
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        quantization_preserving=q_preserving,
        supported_input_activation_n_bits=[2, 4, 8],
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=None,
        signedness=Signedness.AUTO
    )

    a_qcfg = NodeActivationQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                              activation_quantization_fn=None,
                                              activation_quantization_params_fn=None)
    if len(aq_params) != 0:
        a_qcfg.set_activation_quantization_param(aq_params)
    w_qcfg = NodeWeightsQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                        weights_channels_axis=None,
                                        node_attrs_list=[])
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg,
                                         weights_quantization_cfg=w_qcfg)
    return qc

# test graph
def get_test_graph():
    n1 = build_node('input', layer_class=InputLayer)
    conv1 = build_node('conv1', layer_class=Conv2D, qcs=[build_qc(aq_params={'threshold': 8.0, 'is_signed': False})])
    conv2 = build_node('conv2', layer_class=Conv2D, qcs=[build_qc(aq_params={'threshold': 8.0, 'is_signed': False})])
    dropout = build_node('dropout', layer_class=Dropout, qcs=[build_qc(a_enable=False, q_preserving=True)])
    flatten = build_node('flatten', layer_class=Flatten, qcs=[build_qc(a_enable=False, q_preserving=True)])
    fc = build_node('fc', layer_class=Linear, qcs=[build_qc(aq_params={'threshold': 8.0, 'is_signed': False})])

    graph = Graph('g', input_nodes=[n1],
                  nodes=[conv1, conv2, dropout, flatten],
                  output_nodes=[fc],
                  edge_list=[Edge(n1, conv1, 0, 0),
                             Edge(conv1, conv2, 0, 0),
                             Edge(conv2, dropout, 0, 0),
                             Edge(dropout, flatten, 0, 0),
                             Edge(flatten, fc, 0, 0),
                             ]
                  )
    return graph

# test case for get_activation_quantizer_holder in quantization preserving
test_input_0 = ("conv1",)
test_input_1 = ("conv2",)
test_input_2 = ("dropout",)
test_input_3 = ("flatten",)
test_input_4 = ("fc",)

test_expected_0 = (PytorchActivationQuantizationHolder,)
test_expected_1 = (PytorchActivationQuantizationHolder,)
test_expected_2 = (PytorchPreservingActivationQuantizationHolder,)
test_expected_3 = (PytorchPreservingActivationQuantizationHolder,)
test_expected_4 = (PytorchActivationQuantizationHolder,)

@pytest.mark.parametrize(("inputs", "expected"), [
    (test_input_0, test_expected_0),
    (test_input_1, test_expected_1),
    (test_input_2, test_expected_2),
    (test_input_3, test_expected_3),
    (test_input_4, test_expected_4),
])
def test_get_activation_quantizer_holder_quantization_preserving(inputs, expected):

    graph = get_test_graph()
    node_name = inputs[0]
    node = graph.find_node_by_name(node_name)[0] # get node

    result = get_activation_quantizer_holder(node, fw_impl=fw_impl)

    if node_name == "dropout" or node_name == "flatten":
        assert isinstance(result, expected[0])
        assert result.quantization_bypass == True
    else:
        assert isinstance(result, expected[0])