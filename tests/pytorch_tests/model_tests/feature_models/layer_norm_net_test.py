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
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This tests check the addition and subtraction operations. 
Both with different layers and with constants.
"""
class LayerNormNet(torch.nn.Module):
    def __init__(self):
        super(LayerNormNet, self).__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.weight = torch.nn.Parameter(torch.tensor([1., 1., 1.]))

    def forward(self, x):
        # Transpose the tensor such that last dim is the channels.
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)

        # Apply layer_norm
        x = torch.nn.functional.layer_norm(x, normalized_shape=(3,), weight=self.weight, bias=self.bias) # Layer normalization along the last dimension
        return x



class LayerNormNetTest(BasePytorchTest):
    """
    This tests check the addition and subtraction operations.
    Both with different layers and with constants.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return LayerNormNet()