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
This tests checks a model that has calls to torch.Tensor functions,
such as torch.Tensor.size and torch.Tensor.view.
"""
class TorchTensorAttrNet(torch.nn.Module):
    def __init__(self):
        super(TorchTensorAttrNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x.size(1)
        return x.view(1, -1)


class TorchTensorAttrNetTest(BasePytorchTest):
    """
    This tests checks a model that has calls to torch.Tensor functions,
    such as torch.Tensor.size and torch.Tensor.view.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test, convert_to_fx=False)

    def create_feature_network(self, input_shape):
        return TorchTensorAttrNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        super(TorchTensorAttrNetTest, self).compare(quantized_models, float_model, input_x=input_x, quantization_info=quantization_info)