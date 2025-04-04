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

from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc

fw_impl = PytorchImplementation()

class InputLayer:
    pass
class Conv2D:
    pass
class Flatten:
    pass

# test graph
def get_test_graph(q_preserving):
    n1 = build_node('input', layer_class=InputLayer)
    conv1 = build_node('conv1', layer_class=Conv2D, qcs=[build_nbits_qc(a_nbits=4)])
    flatten = build_node('flatten', layer_class=Flatten, qcs=[build_nbits_qc(a_nbits=8, q_preserving=q_preserving)])

    graph = Graph('g', input_nodes=[n1],
                  nodes=[conv1],
                  output_nodes=[flatten],
                  edge_list=[Edge(n1, conv1, 0, 0),
                             Edge(conv1, flatten, 0, 0),
                             ]
                  )
    return graph

# test case for get_activation_quantizer_holder in quantization preserving
test_input_0 = (True, )
test_input_1 = (False, )

test_expected_0 = (PytorchPreservingActivationQuantizationHolder, 4, 4.0, False)
test_expected_1 = (PytorchActivationQuantizationHolder, 8, 8.0, True)

@pytest.mark.parametrize(("inputs", "expected"), [
    (test_input_0, test_expected_0),
    (test_input_1, test_expected_1),
])
def test_get_activation_quantizer_holder_quantization_preserving(inputs, expected):

    graph = get_test_graph(inputs[0]) # set quantization preserving

    prev_node = graph.find_node_by_name('conv1')[0]
    prev_node.candidates_quantization_cfg[0].activation_quantization_cfg.set_activation_quantization_param({'threshold': 4.0, 'is_signed': False})
    prev_node.final_activation_quantization_cfg = prev_node.candidates_quantization_cfg[0].activation_quantization_cfg

    node = graph.find_node_by_name('flatten')[0]
    node.candidates_quantization_cfg[0].activation_quantization_cfg.set_activation_quantization_param({'threshold': 8.0, 'is_signed': True})
    node.final_activation_quantization_cfg = node.candidates_quantization_cfg[0].activation_quantization_cfg
    
    result = get_activation_quantizer_holder(node, prev_node, fw_impl=fw_impl)

    assert isinstance(result, expected[0])
    if isinstance(result, PytorchPreservingActivationQuantizationHolder):
        assert result.quantization_bypass == True
    
    assert result.activation_holder_quantizer.num_bits == expected[1]
    assert result.activation_holder_quantizer.threshold_np == expected[2]
    assert result.activation_holder_quantizer.signed == expected[3]