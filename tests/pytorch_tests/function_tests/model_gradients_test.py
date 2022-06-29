# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.pytorch.back2framework.model_gradients import PytorchModelGradients
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tp_model import get_op_quantization_configs, \
    generate_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tpc_pytorch import generate_pytorch_tpc
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the BatchNorm info collection.
"""


def bn_weight_change(bn: torch.nn.Module):
    bw_shape = bn.weight.shape
    delattr(bn, 'weight')
    delattr(bn, 'bias')
    delattr(bn, 'running_var')
    delattr(bn, 'running_mean')
    bn.register_buffer('weight', torch.rand(bw_shape))
    bn.register_buffer('bias', torch.rand(bw_shape))
    bn.register_buffer('running_var', torch.abs(torch.rand(bw_shape)))
    bn.register_buffer('running_mean', torch.rand(bw_shape))
    return bn


class create_model_1(torch.nn.Module):
    def __init__(self):
        super(create_model_1, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.bn = bn_weight_change(self.bn)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        return x + inp


class create_model_2(torch.nn.Module):
    def __init__(self):
        super(create_model_2, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.bn = bn_weight_change(self.bn)
        self.bn2 = BatchNorm2d(3)
        self.bn2 = bn_weight_change(self.bn2)

    def forward(self, inp):
        x = self.conv1(inp)
        x2 = self.bn(x)
        y = self.bn2(x)
        return x2 + y + inp


class create_model_3(torch.nn.Module):
    def __init__(self):
        super(create_model_3, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.bn = bn_weight_change(self.bn)
        self.bn2 = BatchNorm2d(3)
        self.bn2 = bn_weight_change(self.bn2)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.bn2(x)
        return x + inp


class create_model_4(torch.nn.Module):
    def __init__(self):
        super(create_model_4, self).__init__()
        self.bn = BatchNorm2d(3)
        self.bn = bn_weight_change(self.bn)
        self.bn2 = BatchNorm2d(3)
        self.bn2 = bn_weight_change(self.bn2)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.bn2(x)
        return x + inp


class create_model_5(torch.nn.Module):
    def __init__(self):
        super(create_model_5, self).__init__()
        self.bn = BatchNorm2d(3)
        self.bn = bn_weight_change(self.bn)
        self.relu1 = ReLU()
        self.bn2 = BatchNorm2d(3)
        self.bn2 = bn_weight_change(self.bn2)
        self.bn3 = BatchNorm2d(3)
        self.bn3 = bn_weight_change(self.bn3)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu1(x)
        x = self.bn2(x)
        x = x + inp
        x = self.bn3(x)
        return x + inp


class create_model_6(torch.nn.Module):
    def __init__(self):
        super(create_model_6, self).__init__()
        self.bn = BatchNorm2d(3)
        self.bn = bn_weight_change(self.bn)
        self.bn2 = BatchNorm2d(3)
        self.bn2 = bn_weight_change(self.bn2)

    def forward(self, inp):
        x = self.bn(inp)
        x2 = self.bn2(inp)
        return x2 + x + inp


class ModelGradientsTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        inputs = []
        for in_shape in input_shapes:
            t = torch.randn(*in_shape)
            t.requires_grad_()
            inputs.append(t)
        inputs = to_torch_tensor(inputs)
        return inputs

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        return self.generate_inputs(input_shapes)

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        qc = DEFAULTCONFIG
        pytorch_impl = PytorchImplementation()

        graph = pytorch_impl.model_reader(in_model, self.representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(qc))

        base_config, op_cfg_list = get_op_quantization_configs()
        tp = generate_tp_model(base_config, base_config, op_cfg_list, "model_grad_test")
        tpc = generate_pytorch_tpc(name="model_grad_test", tp_model=tp)

        graph.set_fw_info(fw_info)
        graph.set_tpc(tpc)

        graph = set_quantization_configuration_to_graph(graph=graph,
                                                        quant_config=qc)
        return graph

    def run_test(self, seed=0):
        model_float = create_model_1()
        graph = self.prepare_graph(model_float)

        model_grads = PytorchModelGradients(graph_float=graph,
                                            model_input_tensors=None,
                                            interest_points=[n for n in graph.get_topo_sorted_nodes()],
                                            output_list=None,
                                            all_outputs_indices=None)
        input_tensors = self.representative_data_gen()
        # for t in input_tensors:
        #     t.required_grad = True
        output = model_grads.forward(input_tensors)
        loss = torch.sum(output[0])
        loss.backward()
        # print(model_grads.grads)
        for t in output:
            print(t.grad)
        # self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('conv1_bn')) == 1)
        # conv_bn_node = transformed_graph.find_node_by_name('conv1_bn')[0]
        #
        # prior_std = conv_bn_node.prior_info.std_output
        # prior_mean = conv_bn_node.prior_info.mean_output
        #
        # bn_layer = model_float.bn
        # gamma = bn_layer.weight
        # beta = bn_layer.bias
        #
        # self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == prior_std).all())
        # self.unit_test.assertTrue((beta.cpu().data.numpy() == prior_mean).all())
