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
from torch.nn.functional import softmax

from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_tp_model
from model_compression_toolkit.core.pytorch.utils import set_model
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the Softmax shift feature.
"""


class SoftmaxBaseTest(BasePytorchTest):
    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='8bit_softmax_shift',
                                         ftp_name='softmax_shift_pytorch_test')

    def get_quantization_configs(self):
        quant_config = self.get_quantization_config()
        quant_config.softmax_shift = True
        return {"8bit_softmax_shift": quant_config}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quant_bias = quantized_model.linear.bias
            float_bias = float_model.linear.bias
            diff_bias = float_bias - quant_bias
            diff_bias = diff_bias.detach().cpu()
            mean_diff_bias = diff_bias.mean()

            self.unit_test.assertTrue(np.allclose(diff_bias, mean_diff_bias, atol=1e-1))

            no_softmax_quantized_output = quantized_model(input_x[0])[0]
            no_softmax_float_output = float_model(input_x[0])[0]
            diff_output = no_softmax_float_output - no_softmax_quantized_output
            diff_output = diff_output.detach().cpu()
            mean_diff_output = diff_output.mean()

            self.unit_test.assertTrue(np.allclose(mean_diff_output, mean_diff_bias, atol=1e-1))


class SoftmaxLayerNet(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLayerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.linear = torch.nn.Linear(3*32*32, 3)
        self.softmax = torch.nn.Softmax()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.linear(torch.flatten(x, 1))
        x2 = self.softmax(x)
        return x, x2


class SoftmaxFunctionNet(torch.nn.Module):
    def __init__(self):
        super(SoftmaxFunctionNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.linear = torch.nn.Linear(3 * 32 * 32, 3)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.linear(torch.flatten(x, 1))
        x2 = softmax(x)
        return x, x2


class SoftmaxLayerNetTest(SoftmaxBaseTest):

    """
    This test checks the Softmax shift feature with Softmax as layer.
    """
    def create_feature_network(self, input_shape):
        return SoftmaxLayerNet()


class SoftmaxFunctionNetTest(SoftmaxBaseTest):

    """
    This test checks the Softmax shift feature with Softmax as function.
    """
    def create_feature_network(self, input_shape):
        return SoftmaxFunctionNet()