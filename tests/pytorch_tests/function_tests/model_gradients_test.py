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
from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
import numpy as np

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit.core.common.hessian as hessian_common

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


class basic_derivative_model(torch.nn.Module):
    def __init__(self):
        super(basic_derivative_model, self).__init__()

    def forward(self, inp):
        x = torch.mul(inp, 2)
        x = x + 1
        return x


class basic_model(torch.nn.Module):
    def __init__(self):
        super(basic_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        return x


class advanced_model(torch.nn.Module):
    def __init__(self):
        super(advanced_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn1 = BatchNorm2d(3)
        self.relu1 = ReLU()

        self.conv2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn2 = BatchNorm2d(3)
        self.relu2 = ReLU()

        self.bn3 = BatchNorm2d(3)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = torch.reshape(x, [-1])
        x = self.bn3(x)
        return x


class multiple_output_model(torch.nn.Module):
    def __init__(self):
        super(multiple_output_model, self).__init__()
        self.linear = Linear(32, 32)
        self.conv1 = Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()
        self.conv2 = Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inp):
        x = self.linear(inp)
        x = self.conv1(x)
        x = self.bn(x)
        x1 = self.relu(x)
        x2 = self.conv2(x1)
        return x1, x2


class model_with_output_replacements(torch.nn.Module):
    def __init__(self):
        super(model_with_output_replacements, self).__init__()

        self.conv = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv(inp)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.argmax(x)
        return x


class node_with_multiple_outputs_model(torch.nn.Module):
    def __init__(self):
        super(node_with_multiple_outputs_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        y, z, w = torch.split(x, split_size_or_sections=1, dim=1)
        return y, z, w


class non_differentiable_node_model(torch.nn.Module):
    def __init__(self):
        super(non_differentiable_node_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        y = torch.argmax(x)  # this is a dummy operation just to cover edge case in the gradients computation

        return x * y


def generate_inputs(inputs_shape):
    inputs = []
    for in_shape in inputs_shape:
        t = torch.randn(*in_shape)
        t.requires_grad_()
        inputs.append(t)
    inputs = to_torch_tensor(inputs)
    return inputs


class ModelGradientsCalculationTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self, n_iters=1):
        input_shapes = self.create_inputs_shape()
        for _ in range(n_iters):
            yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0):
        model_float = basic_derivative_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [n for n in graph.get_topo_sorted_nodes()]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0)

        self.unit_test.assertTrue(np.isclose(model_grads[0], 0.8, 1e-1))
        self.unit_test.assertTrue(np.isclose(model_grads[1], 0.2, 1e-1))
        self.unit_test.assertTrue(model_grads[2] == 0.0)



def _get_normalized_hessian_trace_approx(representative_data_gen,
                                         graph,
                                         interest_points,
                                         pytorch_impl,
                                         alpha):

    hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                        representative_dataset=representative_data_gen,
                                                        fw_impl=pytorch_impl)
    x = []
    for interest_point in interest_points:
        request = hessian_common.TraceHessianRequest(mode=hessian_common.HessianMode.ACTIVATION,
                                                     granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                                     target_node=interest_point)
        hessian_data = hessian_service.fetch_hessian(request, 1)
        hessian_data_per_image = hessian_data[0]
        assert isinstance(hessian_data_per_image, list)
        assert len(hessian_data_per_image) == 1
        x.append(hessian_data_per_image[0])
    x = hessian_common.hessian_utils.normalize_weights(x, alpha=alpha, outputs_indices=[len(interest_points) - 1])
    return x


class ModelGradientsBasicModelTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)


    def run_test(self, seed=0):
        model_float = basic_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [n for n in graph.get_topo_sorted_nodes()]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0.3)

        # Checking that the weights where computed and normalized correctly
        self.unit_test.assertTrue(np.isclose(np.sum(model_grads), 1))


class ModelGradientsAdvancedModelTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0):
        model_float = basic_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [n for n in graph.get_topo_sorted_nodes()]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0.3)

        # Checking that the weights where computed and normalized correctly
        self.unit_test.assertTrue(np.isclose(np.sum(model_grads), 1))


class ModelGradientsMultipleOutputsTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0):
        model_float = multiple_output_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [n for n in graph.get_topo_sorted_nodes()]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0.3)

        # Checking that the weights where computed and normalized correctly
        self.unit_test.assertTrue(np.isclose(np.sum(model_grads), 1))


class ModelGradientsOutputReplacementTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0):
        model_float = model_with_output_replacements()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)
        ipts = [n for n in graph.get_topo_sorted_nodes()]
        with self.unit_test.assertRaises(Exception) as e:
            _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                 graph=graph,
                                                 interest_points=ipts,
                                                 pytorch_impl=pytorch_impl,
                                                 alpha=0.3)
        self.unit_test.assertTrue("All graph outputs should support metric outputs" in str(e.exception))


class ModelGradientsMultipleOutputsModelTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        model_float = node_with_multiple_outputs_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [n for n in graph.get_topo_sorted_nodes()]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0.3)

        # Checking that the weights where computed and normalized correctly
        self.unit_test.assertTrue(np.isclose(np.sum(model_grads), 1))


class ModelGradientsNonDifferentiableNodeModelTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        model_float = non_differentiable_node_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [n for n in graph.get_topo_sorted_nodes()]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0.3)

        # Checking that the weights where computed and normalized correctly
        self.unit_test.assertTrue(np.isclose(np.sum(model_grads), 1))


class ModelGradientsSinglePointTest(ModelGradientsBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self, seed=0):
        model_float = basic_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        ipts = [graph.get_topo_sorted_nodes()[-1]]
        model_grads = _get_normalized_hessian_trace_approx(representative_data_gen=self.representative_data_gen,
                                                           graph=graph,
                                                           interest_points=ipts,
                                                           pytorch_impl=pytorch_impl,
                                                           alpha=0.3)

        self.unit_test.assertTrue(len(model_grads) == 1 and model_grads[0] == 1.0)