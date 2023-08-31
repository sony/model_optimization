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

from model_compression_toolkit.core import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc


class argmax_output_model(torch.nn.Module):
    def __init__(self):
        super(argmax_output_model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=(5, 5))
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        output = torch.argmax(x, dim=-1)
        return output


class softmax_output_model(torch.nn.Module):
    def __init__(self):
        super(softmax_output_model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=(5, 5))
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        output = torch.softmax(x, dim=-1)
        return output


class TestSensitivityEvalWithOutputReplacementBase(BasePytorchTest):
    def create_inputs_shape(self):
        return [[1, 3, 16, 16]]

    def representative_data_gen(self, n_iters=1):
        input_shapes = self.create_inputs_shape()
        for _ in range(n_iters):
            yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        raise NotImplementedError("This is a tests base class which do not implement run_test method.")

    def verify_test_for_model(self, model):
        pytorch_impl = PytorchImplementation()
        graph = prepare_graph_with_quantization_parameters(model,
                                                           pytorch_impl,
                                                           DEFAULT_PYTORCH_INFO,
                                                           self.representative_data_gen,
                                                           generate_pytorch_tpc,
                                                           input_shape=(1, 3, 16, 16),
                                                           mixed_precision_enabled=True)

        se = pytorch_impl.get_sensitivity_evaluator(graph,
                                                    MixedPrecisionQuantizationConfigV2(use_grad_based_weights=True),
                                                    self.representative_data_gen,
                                                    DEFAULT_PYTORCH_INFO)

        # If the output replacement nodes for MP sensitivity evaluation has been computed correctly then the ReLU layer
        # should be added to the interest points and included in the output nodes list for metric computation purposes.
        relu_node = graph.get_topo_sorted_nodes()[-2]

        self.unit_test.assertTrue(relu_node.type == torch.nn.ReLU)
        self.unit_test.assertIn(relu_node, se.interest_points)
        self.unit_test.assertEqual(len(se.outputs_replacement_nodes), 1)
        self.unit_test.assertIn(relu_node, se.outputs_replacement_nodes)
        self.unit_test.assertEqual(se.output_nodes_indices, [2, 3])


class TestSensitivityEvalWithArgmaxOutputReplacementNodes(TestSensitivityEvalWithOutputReplacementBase):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self, seed=0, **kwargs):
        model = argmax_output_model()
        self.verify_test_for_model(model)


class TestSensitivityEvalWithSoftmaxOutputReplacementNodes(TestSensitivityEvalWithOutputReplacementBase):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self, seed=0, **kwargs):
        model = softmax_output_model()
        self.verify_test_for_model(model)
