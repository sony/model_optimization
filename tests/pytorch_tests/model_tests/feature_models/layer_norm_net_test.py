# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This tests check the addition and subtraction operations. 
Both with different layers and with constants.
"""
class LayerNormNet(torch.nn.Module):
    def __init__(self, has_weight=None, has_bias=None):
        super(LayerNormNet, self).__init__()

        self.has_weight = has_weight
        self.has_bias = has_bias
        self.bias0 = torch.nn.Parameter(torch.rand(3))
        self.weight0 = torch.nn.Parameter(torch.rand(3))
        self.bias1 = torch.nn.Parameter(torch.rand(3))
        self.weight1 = torch.nn.Parameter(torch.rand(3))

    def forward(self, x):
        # Transpose the tensor such that last dim is the channels.
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)

        # Apply layer_norm with all the combinations of arguments.
        if self.has_weight and self.has_bias:
            x = torch.nn.functional.layer_norm(x, normalized_shape=(3,), weight=self.weight0, bias=self.bias0)
        elif self.has_weight and not self.has_bias:
            x = torch.nn.functional.layer_norm(x, normalized_shape=(3,), weight=self.weight1) # Layer normalization along the last dimension
        elif not self.has_weight and self.has_bias:
            x = torch.nn.functional.layer_norm(x, normalized_shape=(3,), bias=self.bias1)  # Layer normalization along the last dimension
        else:
            x = torch.nn.functional.layer_norm(x, normalized_shape=(3,))  # Layer normalization along the last dimension

        return x



class LayerNormNetTest(BasePytorchTest):
    """
    This tests check the addition and subtraction operations.
    Both with different layers and with constants.
    """
    def __init__(self, unit_test, has_weight=None, has_bias=None):
        super().__init__(unit_test)
        self.has_weight = has_weight
        self.has_bias = has_bias

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return LayerNormNet(self.has_weight, self.has_bias)