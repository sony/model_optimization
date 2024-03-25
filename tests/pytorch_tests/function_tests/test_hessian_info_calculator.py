# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear, Hardswish

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
import numpy as np

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit.core.common.hessian as hessian_common

"""
This test checks the model gradients computation
"""


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
        self.dense = Linear(8, 7)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dense(x)
        return x


class multiple_outputs_model(torch.nn.Module):
    def __init__(self):
        super(multiple_outputs_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn1 = BatchNorm2d(3)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn2 = BatchNorm2d(3)
        self.hswish = Hardswish()
        self.dense = Linear(8, 7)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x1 = self.relu1(x)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x3 = self.hswish(x2)
        x3 = self.dense(x3)
        return x1, x2, x3


class reused_model(torch.nn.Module):
    def __init__(self):
        super(reused_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn1 = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x_split = torch.split(x1, split_size_or_sections=4, dim=-1)
        x1 = self.conv1(x_split[0])
        x2 = x_split[1]
        x1 = self.relu(x1)
        y = torch.concat([x1, x2], dim=-1)
        return y


def generate_inputs(inputs_shape):
    inputs = []
    for in_shape in inputs_shape:
        t = torch.randn(*in_shape)
        t.requires_grad_()
        inputs.append(t)
    inputs = to_torch_tensor(inputs)
    return inputs


def get_expected_shape(t_shape, granularity):
    if granularity == hessian_common.HessianInfoGranularity.PER_ELEMENT:
        return t_shape
    elif granularity == hessian_common.HessianInfoGranularity.PER_TENSOR:
        return (1,)
    else:
        return (t_shape[0],)


class BaseHessianTraceBasicModelTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 8, 8]]

    @staticmethod
    def generate_inputs(input_shapes):
        return generate_inputs(input_shapes)

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def test_hessian_trace_approx(self,
                                  hessian_service,
                                  interest_point,
                                  mode,
                                  granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                  num_scores=1):
        request = hessian_common.TraceHessianRequest(mode=mode,
                                                     granularity=granularity,
                                                     target_node=interest_point)
        expected_shape = get_expected_shape(interest_point.weights['weight'].shape, granularity)
        info = hessian_service.fetch_hessian(request, num_scores)
        assert len(info) == num_scores, f"fetched {num_scores} score but {len(info)} scores were fetched"
        score = np.mean(np.stack(info), axis=0)

        self.unit_test.assertTrue(isinstance(info, list))
        self.unit_test.assertTrue(len(info) == num_scores,
                                  f"fetched {num_scores} score but {len(info)} scores were fetched")
        self.unit_test.assertTrue(score.shape == expected_shape,
                                  f"Tensor shape is expected to be {expected_shape} but has shape {score.shape}")

    def _setup(self):
        model_float = basic_model()
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, generate_pytorch_tpc)

        return graph, pytorch_impl


class WeightsHessianTraceBasicModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                                           mode=hessian_common.HessianMode.WEIGHTS)


class WeightsHessianTraceAdvanceModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 2

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=1,
                                           granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=2,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=3,
                                           granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                                           mode=hessian_common.HessianMode.WEIGHTS)


class WeightsHessianTraceMultipleOutputsModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=1,
                                           granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=2,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=3,
                                           granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                                           mode=hessian_common.HessianMode.WEIGHTS)


class WeightsHessianTraceReuseModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=1,
                                           granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=2,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.WEIGHTS)
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=3,
                                           granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                                           mode=hessian_common.HessianMode.WEIGHTS)


class ActivationHessianTraceBasicModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.ACTIVATION)


class ActivationHessianTraceAdvanceModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 2

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=2,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.ACTIVATION)


class ActivationHessianTraceMultipleOutputsModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=2,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.ACTIVATION)


class ActivationHessianTraceReuseModelTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)
        ipts = [n for n in graph.get_topo_sorted_nodes() if len(n.weights) > 0]
        for ipt in ipts:
            self.test_hessian_trace_approx(hessian_service,
                                           interest_point=ipt,
                                           num_scores=2,
                                           granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                           mode=hessian_common.HessianMode.ACTIVATION)



class ActivationHessianOutputExceptionTest(BaseHessianTraceBasicModelTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.val_batch_size = 1

    def run_test(self, seed=0):
        graph, pytorch_impl = self._setup()
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=self.representative_data_gen,
                                                            fw_impl=pytorch_impl)

        with self.unit_test.assertRaises(Exception) as e:
            request = hessian_common.TraceHessianRequest(mode=hessian_common.HessianMode.ACTIVATION,
                                                         granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                                         target_node=graph.get_outputs()[0].node)
            _ = hessian_service.fetch_hessian(request, required_size=1)

        self.unit_test.assertTrue("Activation Hessian approximation cannot be computed for model outputs"
                                  in str(e.exception))