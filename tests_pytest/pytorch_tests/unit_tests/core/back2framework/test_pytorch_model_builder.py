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
from unittest.mock import Mock
from typing import List
import torch

from model_compression_toolkit.core.pytorch.back2framework.mixed_precision_model_builder import \
    MixedPrecisionPyTorchModelBuilder
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder, fully_quantized_wrapper
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder, \
    PytorchQuantizationWrapper

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common import BaseNode
from tests_pytest._test_util.graph_builder_utils import DummyLayer, build_node as util_build_node, build_nbits_qc
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness
from model_compression_toolkit.core import QuantizationConfig
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema


@pytest.fixture
def graph_mock():
    """ Basic Graph mock with basic retrieve_preserved_quantization_node operation for handling non
    quantization preserving nodes. """
    return Mock(spec_set=Graph, nodes=[], retrieve_preserved_quantization_node=lambda x: x)


def build_node(name='node', framework_attr: dict={}, qcs: List[CandidateNodeQuantizationConfig] = None,
               input_shape=(4, 5, 6), output_shape=(4, 5, 6), weights={},
               layer_class: type = None, reuse=False, set_final_act_config=True):

    assert layer_class is not None
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
        if set_final_act_config:
            node.final_activation_quantization_cfg = node.candidates_quantization_cfg[0].activation_quantization_cfg
    return node


def build_qc(a_nbits=8, a_enable=True, q_preserving=False, aq_params={}):
    op_cfg = OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=a_nbits,
        enable_activation_quantization=a_enable,
        quantization_preserving=q_preserving,
        supported_input_activation_n_bits=8,
        signedness=Signedness.AUTO
    )
    a_qcfg = NodeActivationQuantizationConfig(qc=QuantizationConfig(), op_cfg=op_cfg,
                                              activation_quantization_fn=None,
                                              activation_quantization_params_fn=None)
    if len(aq_params) != 0:
        a_qcfg.set_activation_quantization_param(aq_params)
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg)
    return qc


def get_tpc():
    base_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)
    
    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    mixed_precision_cfg_list = [base_config]

    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list), 
                                                                             base_config=base_config)
    operator_set = []
    preserving_quantization_config = (default_configuration_options.clone_and_edit(enable_activation_quantization=False, quantization_preserving=True))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.DROPOUT, qc_options=preserving_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.FLATTEN, qc_options=preserving_quantization_config))
    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED, qc_options=mixed_precision_configuration_options)

    operator_set.extend([conv, fc])
    tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set),
        insert_preserving_quantizers=True)
    return tpc

# test graph
def get_test_graph():

    conv = build_node('conv', framework_attr={'in_channels':3, 'out_channels':3, 'kernel_size':3}, layer_class=torch.nn.Conv2d, qcs=[build_qc()])
    dropout = build_node('dropout', layer_class=torch.nn.Dropout, qcs=[build_qc(a_enable=False, q_preserving=True)])
    flatten1 = build_node('flatten1', layer_class=torch.nn.Flatten, qcs=[build_qc(a_enable=False, q_preserving=True)])
    flatten2 = build_node('flatten2', layer_class=torch.nn.Flatten, qcs=[build_qc(a_enable=False, q_preserving=True)])
    fc1 = build_node('fc1', framework_attr={'in_features':48, 'out_features':128}, layer_class=torch.nn.Linear, qcs=[build_qc()])
    fc2 = build_node('fc2', framework_attr={'in_features':48, 'out_features':128}, layer_class=torch.nn.Linear, qcs=[build_qc()])

    graph = Graph('g', input_nodes=[conv],
                  nodes=[flatten1, fc1, flatten2, dropout],
                  output_nodes=[fc2],
                  edge_list=[Edge(conv, flatten1, 0, 0),
                             Edge(flatten1, fc1, 0, 0),
                             Edge(fc1, flatten2, 0, 0),
                             Edge(flatten2, dropout, 0, 0),
                             Edge(dropout, fc2, 0, 0),
                             ]
                  )
    tpc = get_tpc()
    fqc = FrameworkQuantizationCapabilities(tpc=tpc, name="test")
    graph.set_fqc(fqc)

    return graph

def get_inferable_quantizers_mock(node):
    if node.name == 'conv':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 8
        activation_quantizers.signed = False
        activation_quantizers.threshold_np = 8.0
    
    elif node.name == 'fc1':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 16
        activation_quantizers.signed = True
        activation_quantizers.threshold_np = 16.0
        
    elif node.name == 'fc2':
        activation_quantizers = Mock()
        activation_quantizers.num_bits = 4
        activation_quantizers.signed = False
        activation_quantizers.threshold_np = 4.0
    else:
        return {}, []
    
    return {}, [activation_quantizers]


class TestPyTorchModelBuilder():

    # test case for PyTorchModelBuilder
    def test_pytorch_model(self, fw_impl_mock):
        graph = get_test_graph()
        fw_impl_mock.get_inferable_quantizers.side_effect = lambda node: get_inferable_quantizers_mock(node)
        exportable_model, _ = PyTorchModelBuilder(graph=graph,
                                                  wrapper=lambda n, m:
                                                  fully_quantized_wrapper(n, m,
                                                                          fw_impl=fw_impl_mock),
                                                  get_activation_quantizer_holder_fn=lambda n, holder_type, **kwargs:
                                                  get_activation_quantizer_holder(n, holder_type,
                                                                                  fw_impl=fw_impl_mock, **kwargs)).build_model()
        
        # check conv
        conv_activation_holder_quantizer = exportable_model.conv_activation_holder_quantizer
        assert isinstance(conv_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert conv_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8
        assert conv_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert conv_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 8.0

        # check flatten1 (same conv)
        flatten1_activation_holder_quantizer = exportable_model.flatten1_activation_holder_quantizer
        assert isinstance(flatten1_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
        assert flatten1_activation_holder_quantizer.quantization_bypass == True
        assert flatten1_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8
        assert flatten1_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert flatten1_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 8.0

        # check fc1
        fc1_activation_holder_quantizer = exportable_model.fc1_activation_holder_quantizer
        assert isinstance(fc1_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert fc1_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert fc1_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert fc1_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0

        # check flatten2 (same fc1)
        flatten2_activation_holder_quantizer = exportable_model.flatten2_activation_holder_quantizer
        assert isinstance(flatten2_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
        assert flatten2_activation_holder_quantizer.quantization_bypass == True
        assert flatten2_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert flatten2_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert flatten2_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0
        
        # check dropout (same fc1)
        dropout_activation_holder_quantizer = exportable_model.dropout_activation_holder_quantizer
        assert isinstance(dropout_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
        assert dropout_activation_holder_quantizer.quantization_bypass == True
        assert dropout_activation_holder_quantizer.activation_holder_quantizer.num_bits == 16
        assert dropout_activation_holder_quantizer.activation_holder_quantizer.signed == True
        assert dropout_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 16.0

        # check fc2
        fc2_activation_holder_quantizer = exportable_model.fc2_activation_holder_quantizer
        assert isinstance(fc2_activation_holder_quantizer, PytorchActivationQuantizationHolder)
        assert fc2_activation_holder_quantizer.activation_holder_quantizer.num_bits == 4
        assert fc2_activation_holder_quantizer.activation_holder_quantizer.signed == False
        assert fc2_activation_holder_quantizer.activation_holder_quantizer.threshold_np == 4.0



class TestPytorchMixedPrecisionModelBuilder:

    @pytest.fixture(autouse=True)
    def setup(self):
        weights_attr_cfg_mock_8 = Mock()
        weights_attr_cfg_mock_8.weights_n_bits = 8
        weights_attr_cfg_mock_8.enable_weights_quantization = True
        weights_attr_cfg_mock_8.weights_channels_axis = [1]
        weights_attr_cfg_mock_8.weights_quantization_fn.return_value = torch.randn(1)
        weights_attr_cfg_mock_4 = Mock()
        weights_attr_cfg_mock_4.weights_n_bits = 4
        weights_attr_cfg_mock_4.enable_weights_quantization = True
        weights_attr_cfg_mock_4.weights_channels_axis = [1]
        weights_attr_cfg_mock_4.weights_quantization_fn.return_value = torch.randn(1)
        weights_attr_cfg_mock_2 = Mock()
        weights_attr_cfg_mock_2.weights_n_bits = 2
        weights_attr_cfg_mock_2.enable_weights_quantization = True
        weights_attr_cfg_mock_2.weights_channels_axis = [1]
        weights_attr_cfg_mock_2.weights_quantization_fn.return_value = torch.randn(1)

        a_qc_mock8 = Mock(spec=NodeActivationQuantizationConfig)
        a_qc_mock8.activation_n_bits = 8
        a_qc_mock8.quant_mode = ActivationQuantizationMode.QUANT
        a_qc_mock4 = Mock(spec=NodeActivationQuantizationConfig)
        a_qc_mock4.activation_n_bits = 4
        a_qc_mock4.quant_mode = ActivationQuantizationMode.QUANT

        w_qc_mock8 = Mock(spec=NodeWeightsQuantizationConfig)
        w_qc_mock8.get_attr_config.return_value = weights_attr_cfg_mock_8

        # activation 8
        self.qc_88 = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock8,
                                                     weights_quantization_cfg=w_qc_mock8)
        w_qc_mock8.enable_weights_quantization = False
        self.qc_act_only = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock8,
                                                           weights_quantization_cfg=w_qc_mock8)

        self.no_quant = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock8,
                                                           weights_quantization_cfg=w_qc_mock8)
        w_qc_mock4 = Mock(spec=NodeWeightsQuantizationConfig)
        w_qc_mock4.get_attr_config.return_value = weights_attr_cfg_mock_4
        self.qc_48 = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock8,
                                                     weights_quantization_cfg=w_qc_mock4)
        w_qc_mock2 = Mock(spec=NodeWeightsQuantizationConfig)
        w_qc_mock2.get_attr_config.return_value = weights_attr_cfg_mock_2
        self.qc_28 = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock8,
                                                     weights_quantization_cfg=w_qc_mock2)

        # activation 4
        w_qc_mock8 = Mock(spec=NodeWeightsQuantizationConfig)
        w_qc_mock8.get_attr_config.return_value = weights_attr_cfg_mock_8
        self.qc_84 = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock4,
                                                     weights_quantization_cfg=w_qc_mock8)
        self.qc_44 = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock4,
                                                     weights_quantization_cfg=w_qc_mock4)
        self.qc_24 = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qc_mock4,
                                                     weights_quantization_cfg=w_qc_mock2)



    def test_model_builder_with_configurable_weights_reused_nodes(self):

        mp_candidautes = [self.qc_88, self.qc_48, self.qc_28]
        mp = build_node('mp', qcs=mp_candidautes, output_shape=(None, 5, 10), layer_class=torch.nn.Conv2d, framework_attr={'in_channels': 3, 'out_channels': 3, 'kernel_size': 3}, set_final_act_config=False)
        mp.reuse_group = 'mp'
        mp_reuse = build_node('mp_reuse', qcs=mp_candidautes, output_shape=(None, 24), reuse=True, layer_class=torch.nn.Conv2d, framework_attr={'in_channels': 3, 'out_channels': 3, 'kernel_size': 3}, set_final_act_config=False)
        mp_reuse.reuse_group = 'mp'

        sp = build_node('sp', qcs=[self.qc_act_only], output_shape=(None, 20, 10), layer_class=torch.nn.ReLU, set_final_act_config=False)
        out = build_node('out', qcs=[self.qc_act_only], output_shape=(None, 17), layer_class=torch.nn.ReLU, set_final_act_config=False)

        nodes = [mp, mp_reuse, sp, out]
        for n in nodes:
            n.find_max_candidate_index = lambda: 0
            n.is_activation_quantization_enabled = lambda: False
            n.is_quantization_preserving = lambda: False

        graph = Graph('g', input_nodes=[mp], nodes=nodes, output_nodes=[out],
                      edge_list=[Edge(mp, sp, 0, 0),
                                 Edge(sp, mp_reuse, 0, 0), Edge(mp_reuse, out, 0, 0)])
        graph.fqc = Mock()

        mp_builder = MixedPrecisionPyTorchModelBuilder(graph)
        model, user_info, conf_node2layers = mp_builder.build_model()
        
        assert 'mp_reuse' not in conf_node2layers
        assert isinstance(model, torch.nn.Module)

        reuse_layer_instance = model.__getattr__('mp_reuse')
        assert isinstance(reuse_layer_instance, PytorchQuantizationWrapper)
        conf_quantizer = reuse_layer_instance.weights_quantizers['weight']
        assert isinstance(conf_quantizer, ConfigurableWeightsQuantizer)
        assert len(conf_quantizer.node_q_cfg) == 3


    def test_model_builder_with_configurable_weights_activation_reused_nodes(self):

        mp_candidautes = [self.qc_88, self.qc_48, self.qc_28, self.qc_84, self.qc_44, self.qc_24]
        mp = build_node('mp', qcs=mp_candidautes, output_shape=(None, 5, 10), layer_class=torch.nn.Conv2d, framework_attr={'in_channels': 3, 'out_channels': 3, 'kernel_size': 3}, set_final_act_config=False)
        mp.reuse_group = 'mp'
        mp_reuse = build_node('mp_reuse', qcs=mp_candidautes, output_shape=(None, 24), reuse=True, layer_class=torch.nn.Conv2d, framework_attr={'in_channels': 3, 'out_channels': 3, 'kernel_size': 3}, set_final_act_config=False)
        mp_reuse.reuse_group = 'mp'

        sp = build_node('sp', qcs=[self.qc_act_only], output_shape=(None, 20, 10), layer_class=torch.nn.ReLU, set_final_act_config=False)
        mp2 = build_node('mp2', qcs=mp_candidautes, output_shape=(None, 150), layer_class=torch.nn.Conv2d, framework_attr={'in_channels': 3, 'out_channels': 3, 'kernel_size': 3}, set_final_act_config=False)
        out = build_node('out', qcs=[self.qc_act_only], output_shape=(None, 17), layer_class=torch.nn.ReLU, set_final_act_config=False)

        nodes = [mp, mp_reuse, sp, mp2, out]
        for n in nodes:
            n.find_max_candidate_index = lambda: 0
            if n.name in ['sp', 'out']:
                n.is_activation_quantization_enabled = lambda: False
            else:
                n.is_activation_quantization_enabled = lambda: True
            n.is_quantization_preserving = lambda: False

        graph = Graph('g', input_nodes=[mp], nodes=nodes, output_nodes=[out],
                      edge_list=[Edge(mp, sp, 0, 0),
                                 Edge(sp, mp_reuse, 0, 0), Edge(mp_reuse, mp2, 0, 0),
                                 Edge(mp2, out, 0, 0)])
        graph.fqc = Mock()

        mp_builder = MixedPrecisionPyTorchModelBuilder(graph)
        model, user_info, conf_node2layers = mp_builder.build_model()

        assert isinstance(model, torch.nn.Module)
        assert 'mp_reuse' in conf_node2layers
        assert len(conf_node2layers['mp_reuse']) == 1

        # Verify reused layer weights configurable quantizer
        reuse_layer_instance = model.__getattr__('mp_reuse')
        assert isinstance(reuse_layer_instance, PytorchQuantizationWrapper)
        conf_quantizer = reuse_layer_instance.weights_quantizers['weight']
        assert isinstance(conf_quantizer, ConfigurableWeightsQuantizer)

        # Verify reused layer activation configurable quantizer

        conf_holder_layer = [x for x in conf_node2layers['mp_reuse'] if isinstance(x, PytorchActivationQuantizationHolder)]
        assert len(conf_holder_layer) == 1
        conf_holder_layer = conf_holder_layer[0]
        
        # Note that due to the test setup with mock quantization candidates, the holder layer have duplicate quantizers
        # for candidates with same activation bit but different weight, so we just want to make sure there are at least
        # two, which means that the reused node is set in the mp model as a activation configurable.  
        assert len(conf_holder_layer.activation_holder_quantizer.activation_quantizers) > 1