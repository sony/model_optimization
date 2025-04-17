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
import torch
from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder, fully_quantized_wrapper, get_preserving_activation_quantizer_holder
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
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_tpc
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR

fw_impl = PytorchImplementation()

def build_node(name='node', framework_attr={}, qcs: List[CandidateNodeQuantizationConfig] = None,
               input_shape=(4, 5, 6), output_shape=(4, 5, 6), weights = {},
               layer_class=DummyLayer, reuse=False):

    node = BaseNode(name=name,
                    framework_attr=framework_attr,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    weights=weights,
                    layer_class=layer_class,
                    reuse=reuse)
    if qcs:
        assert isinstance(qcs, list)
        node.candidates_quantization_cfg = qcs
        node.final_weights_quantization_cfg = node.candidates_quantization_cfg[0].weights_quantization_cfg
        node.final_activation_quantization_cfg = node.candidates_quantization_cfg[0].activation_quantization_cfg
    return node

def build_qc(a_nbits=8, a_enable=True, q_preserving=False, aq_params={}):
    
    qc = QuantizationConfig()
    op_cfg = OpQuantizationConfig(
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

def get_tpc():
    base_cfg, mx_cfg_list, default_config = get_op_quantization_configs()
    base_cfg = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                {
                                    KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                .clone_and_edit(enable_weights_quantization=False),
                                    BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                .clone_and_edit(enable_weights_quantization=False),
                                },
                            )

    for i, mx_cfg in enumerate(mx_cfg_list):
        mx_cfg_list[i] = mx_cfg.clone_and_edit(attr_weights_configs_mapping=
                                    {
                                        KERNEL_ATTR: mx_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                    .clone_and_edit(enable_weights_quantization=False),
                                        BIAS_ATTR: mx_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                    .clone_and_edit(enable_weights_quantization=False),
                                    },
                                )

    tpc = generate_tpc(default_config=default_config, base_config=base_cfg, mixed_precision_cfg_list=mx_cfg_list,
                        name='imx500_tpc')
    return tpc

# test graph
def get_test_graph():

    conv1 = build_node('conv1', framework_attr={'in_channels':3, 'out_channels':3, 'kernel_size':3}, layer_class=torch.nn.Conv2d, qcs=[build_qc(a_nbits=8, aq_params={'threshold': 8.0, 'is_signed': False})])
    conv2 = build_node('conv2', framework_attr={'in_channels':3, 'out_channels':3, 'kernel_size':3}, layer_class=torch.nn.Conv2d, qcs=[build_qc(a_nbits=16, aq_params={'threshold': 4.0, 'is_signed': False})])
    dropout = build_node('dropout', layer_class=torch.nn.Dropout, qcs=[build_qc(a_enable=False, q_preserving=True)])
    flatten = build_node('flatten', layer_class=torch.nn.Flatten, qcs=[build_qc(a_enable=False, q_preserving=True)])
    fc = build_node('fc', framework_attr={'in_features':48, 'out_features':128}, layer_class=torch.nn.Linear, qcs=[build_qc(a_nbits=4, aq_params={'threshold': 8.0, 'is_signed': True})])

    graph = Graph('g', input_nodes=[conv1],
                  nodes=[conv2, dropout, flatten],
                  output_nodes=[fc],
                  edge_list=[Edge(conv1, conv2, 0, 0),
                             Edge(conv2, dropout, 0, 0),
                             Edge(dropout, flatten, 0, 0),
                             Edge(flatten, fc, 0, 0),
                             ]
                  )
    tpc = get_tpc()
    fqc = FrameworkQuantizationCapabilities(tpc=tpc, name="test_pytorch_model")
    graph.set_fqc(fqc)

    return graph

class TestPyTorchModelBuilder():

    # test case for PyTorchModelBuilder
    test_input_0 = ("conv1_activation_holder_quantizer",)
    test_input_1 = ("conv2_activation_holder_quantizer",)
    test_input_2 = ("dropout_activation_holder_quantizer",)
    test_input_3 = ("flatten_activation_holder_quantizer",)
    test_input_4 = ("fc_activation_holder_quantizer",)

    test_expected_0 = (PytorchActivationQuantizationHolder, 8, 8.0, False)
    test_expected_1 = (PytorchActivationQuantizationHolder, 16, 4.0, False)
    test_expected_2 = (PytorchPreservingActivationQuantizationHolder, 16, 4.0, False)
    test_expected_3 = (PytorchPreservingActivationQuantizationHolder, 16, 4.0, False)
    test_expected_4 = (PytorchActivationQuantizationHolder, 4, 8.0, True)

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_0, test_expected_0),
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
        (test_input_4, test_expected_4),
    ])
    def test_pytorch_model(self, inputs, expected):
        graph = get_test_graph()
        exportable_model, _ = PyTorchModelBuilder(graph=graph,
                                                wrapper=lambda n, m:
                                                fully_quantized_wrapper(n, m,
                                                                        fw_impl=fw_impl),
                                                get_activation_quantizer_holder_fn=lambda n:
                                                get_activation_quantizer_holder(n,
                                                                                fw_impl=fw_impl),
                                                get_preserving_activation_quantizer_holder_fn=lambda n:
                                                get_preserving_activation_quantizer_holder(n,
                                                                                fw_impl=fw_impl)).build_model()
        
        preserving_activation_holder_quantizer_name = ["dropout_activation_holder_quantizer", "flatten_activation_holder_quantizer"]
        activation_holder_quantizer_name = ["conv1_activation_holder_quantizer", "conv2_activation_holder_quantizer", "fc_activation_holder_quantizer"]
        for name, layer in exportable_model.named_modules():
            if name == inputs[0]:
                if name in preserving_activation_holder_quantizer_name:
                    assert isinstance(layer, expected[0]) # check holder
                    assert layer.quantization_bypass == True
                    assert layer.activation_holder_quantizer.num_bits == expected[1]
                    assert layer.activation_holder_quantizer.threshold_np == expected[2]
                    assert layer.activation_holder_quantizer.signed == expected[3]

                elif name in activation_holder_quantizer_name:
                    assert isinstance(layer, expected[0]) # check holder
                    assert layer.activation_holder_quantizer.num_bits == expected[1]
                    assert layer.activation_holder_quantizer.threshold_np == expected[2]
                    assert layer.activation_holder_quantizer.signed == expected[3]
