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
from torch import _assert
from torch.fx import symbolic_trace

from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.core.pytorch.utils import set_model

"""
This tests checks that the assert operation is being removed from the graph during quantization.
"""


class AssertNet(torch.nn.Module):
    def __init__(self):
        super(AssertNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x, y):
        _assert(x.shape == y.shape, "Inputs should have the same shape")
        x1 = self.conv1(x)
        assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')
        x2 = x1 + 3
        _assert(x2.shape == x1.shape, "Tensors should have the same shape")
        y = self.conv2(y)
        return x2 - y, y - x2, x2 + y


class AssertNetTest(BasePytorchTest):
    """
    This tests checks that the assert operation is being removed from the graph during quantization.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return AssertNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)

        float_fx_model = symbolic_trace(float_model)
        float_node_list = list(float_fx_model.graph.nodes)

        # check for assert nodes in float model
        self.unit_test.assertTrue(_assert in [node.target for node in float_node_list])

        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)

            quantized_node_list = list(quantized_model.graph.nodes)

            # check for assert nodes in quantized model
            self.unit_test.assertFalse(_assert in [node.layer_class for node in quantized_node_list])
