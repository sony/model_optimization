# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import torch.nn as nn
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor

"""
This test checks dynamic output size for nn.ConvTranspose2d.
"""


class ConvTranspose2dDynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.upsample(x, output_size=[224, 224])  # <--- dynamic output_size
        return x


class ConvTranspose2dDynamicNetTest(BasePytorchTest):
    """
    This test checks the addition and subtraction operations.
    Both with different layers and with constants.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        for _, qmodel in quantized_models.items():
            y_float = float_model(in_torch_tensor)
            y_quant = qmodel(in_torch_tensor)
            self.unit_test.assertTrue(y_float.shape == y_quant.shape,
                                      msg=f'Out shape of the quantized model is not as the float model!')

    def create_feature_network(self, input_shape):
        return ConvTranspose2dDynamicNet()
