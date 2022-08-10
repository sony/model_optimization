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
import numpy as np
import torch
from torch.fx import symbolic_trace

from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode


"""
This tests checks that the "broken" node (node without output) is being removed from the graph during quantization.
"""


class BrokenNet(torch.nn.Module):
    def __init__(self, const):
        super(BrokenNet, self).__init__()
        self.const = const

    def forward(self, x, y):
        x1 = x + self.const
        x2 = x + y
        return x2


class BrokenNetTest(BasePytorchTest):
    """
    This tests checks that the "broken" node (node without output) is being removed from the graph during quantization.
    """

    def __init__(self, unit_test, const=3):
        super().__init__(unit_test)
        self.const = const

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return BrokenNet(self.const)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)

        float_fx_model = symbolic_trace(float_model)
        float_node_list = list(float_fx_model.graph.nodes)

        # check for the const of the "broken" node in nodes in the float model
        self.unit_test.assertTrue(np.any([self.const in args for args in [node.args for node in float_node_list]]))

        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)

            quantized_node_list = list(quantized_model.graph.nodes)

            # check for the const of the "broken" node in nodes in the quantized model
            self.unit_test.assertFalse(np.any(
                [self.const in args for args in [node.op_call_args for node in quantized_node_list
                                                 if node.__class__ == FunctionalNode]]))
