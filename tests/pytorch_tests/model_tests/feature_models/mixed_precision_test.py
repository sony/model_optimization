# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import numpy as np

from model_compression_toolkit import MixedPrecisionQuantizationConfig, KPI
from model_compression_toolkit.common.user_info import UserInformation
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct

"""
This test checks the BatchNorm folding feature, plus adding a residual connection.
"""


class MixedPercisionBaseTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_configs(self):
        qc = mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                    mct.ThresholdSelectionMethod.MSE,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=False,
                                    relu_unbound_correction=False,
                                    input_scaling=False)

        return {"mixed_precision_config":
                    MixedPrecisionQuantizationConfig(qc, weights_n_bits=[2, 8, 4], num_of_images=1)}

    # def get_input_shapes(self):
    #     return [[self.val_batch_size, 224, 244, 3]]

    def create_feature_network(self, input_shape):
        return MixedPrecisionNet(input_shape)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError


class MixedPercisionSearchBasic(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [0, 0]).all()
        # for i in range(30):  # quantized per channel
        #     self.unit_test.assertTrue(
        #         np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 4)
        # for i in range(50):  # quantized per channel
        #     self.unit_test.assertTrue(
        #         np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 4)


class MixedPrecisionNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 4, kernel_size=3, stride=1)
        # self.bn1 = torch.nn.BatchNorm2d()
        # self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        # x = self.bn1(x)
        # x = self.conv2(x)
        # output = self.relu(x)
        return x

