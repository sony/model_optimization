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

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the Symmetric activation quantizer.
"""


class SymmetricActivationTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        tp = generate_test_tp_model({
            'activation_quantization_method': QuantizationMethod.SYMMETRIC,
            'activation_n_bits': 2})
        return {'act_2bit': generate_pytorch_tpc(name="symmetric_layer_test", tp_model=tp)}

    def get_quantization_configs(self):
        return {
            'act_2bit': mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                               mct.QuantizationErrorMethod.NOCLIPPING,
                                               shift_negative_activation_correction=True,
                                               shift_negative_ratio=np.inf),
        }

    def create_feature_network(self, input_shape):
        return SymmetricActivationNet(input_shape)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info: UserInformation = None):
        for model_name, quantized_model in quantized_models.items():
            quantized_model_fx = symbolic_trace(quantized_model)

            # check the activations values changed due to the number of bits
            output = quantized_model(input_x).cpu().detach().numpy()
            self.unit_test.assertTrue(len(np.unique(output)) == 4)


class SymmetricActivationNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(SymmetricActivationNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=(5, 5))

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.conv2(x)
        return x
