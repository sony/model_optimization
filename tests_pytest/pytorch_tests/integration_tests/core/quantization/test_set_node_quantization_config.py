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

import model_compression_toolkit as mct
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.quantization.bit_width_config import ManualBitWidthSelection, ManualWeightsBitWidthSelection
from model_compression_toolkit.core import BitWidthConfig, CoreConfig

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from tests_pytest._test_util.graph_builder_utils import build_node

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import \
    FrameworkQuantizationCapabilities, OperationsSetToLayers

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO

import torch
from torch import nn
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_tpc


#TEST_KERNEL = 'kernel'
#TEST_BIAS = 'bias'

### dummy layer classes
"""
class Conv:
    pass
class InputLayer:
    pass
class Add:
    pass
class BatchNormalization:
    pass
class ReLU:
    pass
class Flatten:
    pass
class Dense:
    pass
"""
#from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs

from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict
#from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR

from tests.common_tests.helpers.generate_test_tpc import generate_tpc_with_activation_mp

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.target_platform_capabilities.constants import PYTORCH_KERNEL, BIAS


def get_tpc(kernel_n, bias_n):
    kernel_weights_n_bits = 8 ### [DEBUG0404] 8 ni suruto Error. 16 dato ugoku.
    bias_weights_n_bits = 32
    activation_n_bits = 8

    base_cfg, _, default_config = get_op_quantization_configs()

    """
    base_cfg = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                            {
                                                KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                            .clone_and_edit(weights_n_bits=kernel_weights_n_bits),
                                                BIAS_ATTR: base_cfg.attr_weights_configs_mapping[BIAS_ATTR]
                                            .clone_and_edit(weights_n_bits=bias_weights_n_bits, enable_weights_quantization=True),
                                            },
                                            activation_n_bits=activation_n_bits)
    #"""

    #weights_04_bits = base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}})
    #weights_02_bits = base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 2}})
    #weights_16_bits = base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 16}})

    #mx_cfg_list = [base_cfg, weights_04_bits, weights_02_bits, weights_16_bits]

    mx_cfg_list = [base_cfg]
    for n in [2,4,16]:
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))
        mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={BIAS_ATTR: {WEIGHTS_N_BITS: n}}))

    # [Error] base_cfg have only one qconfig, so bitwidth cannot change to another number.
    tpc = generate_tpc(default_config=default_config, base_config=base_cfg, mixed_precision_cfg_list=mx_cfg_list, name='imx500_tpc_kai')

    #tpc = mct.get_target_platform_capabilities('pytorch', 'default')


    # [Error] default_config don't have qconfig with weights,  so bitwidth cannot manipulate.
    #tpc = generate_tpc(default_config, base_cfg, mx_cfg_list, 'imx500_tpc_kai')
    """
    # [Error] default_configuration_options.quantization_configurations cannot be multiple lists.
    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple(mx_cfg_list), base_config=base_cfg
    )
    tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        tpc_minor_version=None,
        tpc_patch_version=None,
        tpc_platform_type=None,
        operator_set=None,
        fusing_patterns=None,
        add_metadata=False,
        name='imx500_tpc_kai')
    """
    # [Error] base_cfg have only one qconfig, so bitwidth cannot change to another number.
    #tpc = generate_tpc_multiqco(base_cfg, base_cfg, mx_cfg_list, 'imx500_tpc_kai')
    #tpc.default_qco.quantization_configurations = tuple(mx_cfg_list)

    return tpc


# AttributeQuantizationConfig(weights_quantization_method=<QuantizationMethod.SYMMETRIC: 2>, weights_n_bits=16, weights_per_channel_threshold=True, enable_weights_quantization=True, lut_values_bitwidth=None)
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import AttributeQuantizationConfig
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc

### test model

class TestManualWeightsBitwidthSelection:
    def representative_data_gen(self, shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

    def get_float_model(self):
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv_transpose(x)
                x = self.relu(x)
                return x
        return BaseModel()

    def get_test_graph(self, core_config, kernel_n, bias_n):
        """
        # n1 = build_node('input', layer_class=InputLayer)
        conv1 = build_node('conv1', layer_class=nn.Conv2d,
                           canonical_weights={
                               PYTORCH_KERNEL: AttributeQuantizationConfig(weights_n_bits=8),
                               BIAS: AttributeQuantizationConfig(weights_n_bits=32)}
                           )
        # add1 = build_node('add1', layer_class=Add)
        conv2 = build_node('conv2', layer_class=nn.Conv2d,
                           canonical_weights={
                               PYTORCH_KERNEL: AttributeQuantizationConfig(weights_n_bits=8),
                               BIAS: AttributeQuantizationConfig(weights_n_bits=32)}
                           )
        bn1 = build_node('bn1', layer_class=BatchNormalization)
        relu = build_node('relu1', layer_class=nn.ReLU,
                          canonical_weights={
                              PYTORCH_KERNEL: AttributeQuantizationConfig(weights_n_bits=8),
                              BIAS: AttributeQuantizationConfig(weights_n_bits=32)}
                          )
        # add2 = build_node('add2', layer_class=Add)
        flatten = build_node('flatten', layer_class=nn.Flatten)
        fc = build_node('fc', layer_class=nn.Linear)


        graph = Graph('xyz', input_nodes=[conv1],
                      nodes=[conv1, conv2, bn1, relu, flatten],
                      output_nodes=[fc],
                      edge_list=[  # Edge(n1, conv1, 0, 0),
                          # Edge(conv1, add1, 0, 0),
                          # Edge(add1, conv2, 0, 0),
                          Edge(conv1, conv2, 0, 0),
                          Edge(conv2, bn1, 0, 0),
                          Edge(bn1, relu, 0, 0),
                          # Edge(relu, add2, 0, 0),
                          # Edge(add1, add2, 0, 0),
                          Edge(relu, flatten, 0, 0),
                          # Edge(add2, flatten, 0, 0),
                          Edge(flatten, fc, 0, 0),
                      ]
                      )
        """
        float_model = self.get_float_model()
        print("float_model", float_model)
        fw_info = DEFAULT_PYTORCH_INFO

        fw_impl = PytorchImplementation()
        graph = fw_impl.model_reader(float_model,
                                     self.representative_data_gen)
        graph.set_fw_info(fw_info)

        tpc = get_tpc(kernel_n, bias_n)
        attach2pytorch = AttachTpcToPytorch()
        fqc = attach2pytorch.attach(
            tpc, core_config.quantization_config.custom_tpc_opset_to_layer)
        graph.set_fqc(fqc)

        return graph

    # test case for set_manual_activation_bit_width
    test_input_1 = (NodeTypeFilter(nn.Conv2d), 2, PYTORCH_KERNEL)
    test_input_2 = ([NodeTypeFilter(nn.ConvTranspose2d), NodeNameFilter("conv1")], [16], [PYTORCH_KERNEL])
    test_input_3 = ([NodeTypeFilter(nn.ConvTranspose2d), NodeNameFilter("conv1")], [4, 16], [PYTORCH_KERNEL, BIAS])

    test_expected_1 = (NodeTypeFilter, nn.ReLU, 16)
    test_expected_2 = ([NodeTypeFilter, nn.ReLU, 2], [NodeNameFilter, "conv1", 2])
    test_expected_3 = ([NodeTypeFilter, nn.ReLU, 4], [NodeNameFilter, "conv1", 8])

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        print('# test_manual_weights_bitwidth_selection start.')

        print('inputs', inputs)
        print('expected', expected)

        kernel_n = 8
        bias_n = 32
        if PYTORCH_KERNEL in inputs[2]:
            indices = [index for index, value in enumerate(inputs[2]) if value == PYTORCH_KERNEL]
            kernel_n = inputs[1] if type(inputs[2]) != list else inputs[1][indices[0]]
        if BIAS in inputs[2]:
            indices = [index for index, value in enumerate(inputs[2]) if value == BIAS]
            bias_n = inputs[1] if type(inputs[2]) != list else inputs[1][indices[0]]
        print('kernel_n, bias_n', kernel_n, bias_n)
        core_config = CoreConfig()
        graph = self.get_test_graph(core_config, kernel_n, bias_n)
        #graph = get_test_graph()
        print('graph', graph)

        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

        updated_graph = set_quantization_configuration_to_graph(
            graph, core_config.quantization_config, core_config.bit_width_config,
            False, False
        )
        print('------graph---------------------')
        print('0', graph)
        print('1', graph.nodes)
        print('2', graph.nodes.keys())
        """
        for n in graph.nodes:
            print('n', n)
            a = graph.get_weights_configurable_nodes(DEFAULT_PYTORCH_INFO, True)
            b = graph.get_activation_configurable_nodes()
            print('a', a)
            print('b', b)

        ### len(node.candidates_quantization_cfg) de Error.
        for node in updated_graph.nodes:
            print("z", node) #, node.candidates_quantization_cfg
            for ii in range(len(node.candidates_quantization_cfg)):
                print('z4', ii, node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping)
                #print('z4 0', ii, type(node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping))
                for vkey in node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping:
                    #print('z5', vkey, node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping[vkey])
                    cfg = node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping[vkey]
                    print('z5 cfg.weights_n_bits', cfg.weights_n_bits)
        """

        print('------updated graph---------------------')
        print(updated_graph)

        for node in updated_graph.nodes:
            print("z", node) #, node.candidates_quantization_cfg
            for ii in range(len(node.candidates_quantization_cfg)):
                print('z4', ii, node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping)
                #print('z4 0', ii, type(node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping))
                for vkey in node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping:
                    #print('z5', vkey, node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping[vkey])
                    cfg = node.candidates_quantization_cfg[ii].weights_quantization_cfg.attributes_config_mapping[vkey]
                    print('z5 cfg.weights_n_bits', cfg.weights_n_bits)

            """
            for val2 in node.get_node_weights_attributes():
                print("z2", val2, type(val2))
                a = node.weights[val2]
                print('a', a)
            """

        #assert graph == updated_graph


        print('# test_manual_weights_bitwidth_selection end.')

        pass

