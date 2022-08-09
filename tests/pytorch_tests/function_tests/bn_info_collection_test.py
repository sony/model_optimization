# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
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


class BNInfoCollectionTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        graph = pytorch_impl.model_reader(in_model, representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        transformed_graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
        return transformed_graph

    def run_test(self):
        model_float = create_model_1()
        transformed_graph = self.prepare_graph(model_float)

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('conv1_bn')) == 1)
        conv_bn_node = transformed_graph.find_node_by_name('conv1_bn')[0]

        prior_std = conv_bn_node.prior_info.std_output
        prior_mean = conv_bn_node.prior_info.mean_output

        bn_layer = model_float.bn
        gamma = bn_layer.weight
        beta = bn_layer.bias

        self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == prior_std).all())
        self.unit_test.assertTrue((beta.cpu().data.numpy() == prior_mean).all())


class Conv2D2BNInfoCollectionTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        graph = pytorch_impl.model_reader(in_model, representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        transformed_graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
        return transformed_graph

    def run_test(self):
        model_float = create_model_2()
        transformed_graph = self.prepare_graph(model_float)

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('conv1')) == 1)
        conv_bn_node = transformed_graph.find_node_by_name('conv1')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        conv_std = conv_bn_node.prior_info.std_output
        conv_mean = conv_bn_node.prior_info.mean_output

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        bn_layer = model_float.bn
        mm = bn_layer.running_mean
        mv = bn_layer.running_var
        m_std = np.sqrt(mv.cpu().data.numpy())
        self.unit_test.assertTrue((mm.cpu().data.numpy() == conv_mean).all())
        self.unit_test.assertTrue((m_std == conv_std).all())

        gamma = bn_layer.weight
        beta = bn_layer.bias
        self.unit_test.assertTrue((beta.cpu().data.numpy() == bn_mean).all())
        self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == bn_std).all())

        bn2_layer = model_float.bn2
        gamma2 = bn2_layer.weight
        beta2 = bn2_layer.bias
        self.unit_test.assertTrue((beta2.cpu().data.numpy() == bn2_mean).all())
        self.unit_test.assertTrue((abs(gamma2.cpu().data.numpy()) == bn2_std).all())


class Conv2DBNChainInfoCollectionTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        graph = pytorch_impl.model_reader(in_model, representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        transformed_graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
        return transformed_graph

    def run_test(self):
        model_float = create_model_3()
        transformed_graph = self.prepare_graph(model_float)

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('conv1_bn')) == 1)
        conv_bn_node = transformed_graph.find_node_by_name('conv1_bn')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        prior_std = conv_bn_node.prior_info.std_output
        prior_mean = conv_bn_node.prior_info.mean_output

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        bn_layer = model_float.bn
        gamma = bn_layer.weight
        beta = bn_layer.bias
        self.unit_test.assertTrue((beta.cpu().data.numpy() == prior_mean).all())
        self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == prior_std).all())

        bn2_layer = model_float.bn2
        gamma2 = bn2_layer.weight
        beta2 = bn2_layer.bias
        self.unit_test.assertTrue((beta2.cpu().data.numpy() == bn2_mean).all())
        self.unit_test.assertTrue((abs(gamma2.cpu().data.numpy()) == bn2_std).all())


class BNChainInfoCollectionTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        graph = pytorch_impl.model_reader(in_model, representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        transformed_graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
        return transformed_graph

    def run_test(self):
        model_float = create_model_4()
        transformed_graph = self.prepare_graph(model_float)

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('inp')) == 1)
        inp_node = transformed_graph.find_node_by_name('inp')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        prior_std = inp_node.prior_info.std_output
        prior_mean = inp_node.prior_info.mean_output

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        bn_layer = model_float.bn
        mm = bn_layer.running_mean
        mv = bn_layer.running_var
        m_std = np.sqrt(mv.cpu().data.numpy())
        self.unit_test.assertTrue((mm.cpu().data.numpy() == prior_mean).all())
        self.unit_test.assertTrue((m_std == prior_std).all())

        gamma = bn_layer.weight
        beta = bn_layer.bias
        self.unit_test.assertTrue((beta.cpu().data.numpy() == bn_mean).all())
        self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == bn_std).all())

        bn2_layer = model_float.bn2
        gamma2 = bn2_layer.weight
        beta2 = bn2_layer.bias
        self.unit_test.assertTrue((beta2.cpu().data.numpy() == bn2_mean).all())
        self.unit_test.assertTrue((abs(gamma2.cpu().data.numpy()) == bn2_std).all())


class BNLayerInfoCollectionTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        graph = pytorch_impl.model_reader(in_model, representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        transformed_graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
        return transformed_graph

    def run_test(self):
        model_float = create_model_5()
        transformed_graph = self.prepare_graph(model_float)

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('inp')) == 1)
        inp_node = transformed_graph.find_node_by_name('inp')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('relu1')) == 1)
        relu_node = transformed_graph.find_node_by_name('relu1')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('add')) == 1)
        add_node = transformed_graph.find_node_by_name('add')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn3')) == 1)
        bn3_node = transformed_graph.find_node_by_name('bn3')[0]

        prior_std = inp_node.prior_info.std_output
        prior_mean = inp_node.prior_info.mean_output

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        relu_std = relu_node.prior_info.std_output
        relu_mean = relu_node.prior_info.mean_output

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        add_std = add_node.prior_info.std_output
        add_mean = add_node.prior_info.mean_output

        bn3_std = bn3_node.prior_info.std_output
        bn3_mean = bn3_node.prior_info.mean_output

        bn_layer = model_float.bn
        mm = bn_layer.running_mean
        mv = bn_layer.running_var
        m_std = np.sqrt(mv.cpu().data.numpy())
        self.unit_test.assertTrue((mm.cpu().data.numpy() == prior_mean).all())
        self.unit_test.assertTrue((m_std == prior_std).all())

        gamma = bn_layer.weight
        beta = bn_layer.bias
        self.unit_test.assertTrue((beta.cpu().data.numpy() == bn_mean).all())
        self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == bn_std).all())

        bn2_layer = model_float.bn2
        mm2 = bn2_layer.running_mean
        mv2 = bn2_layer.running_var
        m_std2 = np.sqrt(mv2.cpu().data.numpy())
        self.unit_test.assertTrue((mm2.cpu().data.numpy() == relu_mean).all())
        self.unit_test.assertTrue((m_std2 == relu_std).all())

        gamma2 = bn2_layer.weight
        beta2 = bn2_layer.bias
        self.unit_test.assertTrue((beta2.cpu().data.numpy() == bn2_mean).all())
        self.unit_test.assertTrue((abs(gamma2.cpu().data.numpy()) == bn2_std).all())

        bn3_layer = model_float.bn3
        mm3 = bn3_layer.running_mean
        mv3 = bn3_layer.running_var
        m_std3 = np.sqrt(mv3.cpu().data.numpy())
        self.unit_test.assertTrue((mm3.cpu().data.numpy() == add_mean).all())
        self.unit_test.assertTrue((m_std3 == add_std).all())

        gamma3 = bn3_layer.weight
        beta3 = bn3_layer.bias
        self.unit_test.assertTrue((beta3.cpu().data.numpy() == bn3_mean).all())
        self.unit_test.assertTrue((abs(gamma3.cpu().data.numpy()) == bn3_std).all())


class INP2BNInfoCollectionTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def prepare_graph(self, in_model):
        fw_info = DEFAULT_PYTORCH_INFO
        pytorch_impl = PytorchImplementation()
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        graph = pytorch_impl.model_reader(in_model, representative_data_gen)  # model reading
        graph = substitute(graph, pytorch_impl.get_substitutions_prepare_graph())
        for node in graph.nodes:
            node.prior_info = pytorch_impl.get_node_prior_info(node=node,
                                                               fw_info=fw_info,
                                                               graph=graph)
        transformed_graph = substitute(graph, pytorch_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
        return transformed_graph

    def run_test(self):
        model_float = create_model_6()
        transformed_graph = self.prepare_graph(model_float)

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('inp')) == 1)
        inp_node = transformed_graph.find_node_by_name('inp')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn')[0]

        self.unit_test.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        prior_std = inp_node.prior_info.std_output
        prior_mean = inp_node.prior_info.mean_output

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        bn_layer = model_float.bn
        mm = bn_layer.running_mean
        mv = bn_layer.running_var
        m_std = np.sqrt(mv.cpu().data.numpy())
        self.unit_test.assertTrue((mm.cpu().data.numpy() == prior_mean).all())
        self.unit_test.assertTrue((m_std == prior_std).all())

        gamma = bn_layer.weight
        beta = bn_layer.bias
        self.unit_test.assertTrue((beta.cpu().data.numpy() == bn_mean).all())
        self.unit_test.assertTrue((abs(gamma.cpu().data.numpy()) == bn_std).all())

        bn2_layer = model_float.bn2
        gamma2 = bn2_layer.weight
        beta2 = bn2_layer.bias
        self.unit_test.assertTrue((beta2.cpu().data.numpy() == bn2_mean).all())
        self.unit_test.assertTrue((abs(gamma2.cpu().data.numpy()) == bn2_std).all())
