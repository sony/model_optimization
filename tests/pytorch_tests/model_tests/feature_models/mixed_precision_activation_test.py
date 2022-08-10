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
import numpy as np

from model_compression_toolkit import MixedPrecisionQuantizationConfig, KPI
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct
from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict

"""
This test checks the Mixed Precision feature.
"""


class MixedPercisionActivationBaseTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)]),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

    def get_quantization_configs(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=False,
                                    relu_bound_to_power_of_2=False,
                                    input_scaling=False)

        return {"mixed_precision_activation_model": MixedPrecisionQuantizationConfig(qc, num_of_images=1)}

    def create_feature_network(self, input_shape):
        return MixedPrecisionNet(input_shape)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def verify_config(self, result_config, expected_config):
        # TODO: Add aditional test that maybe check the actual bitwidth (when refactoring MP tests)
        self.unit_test.assertTrue(all(result_config == expected_config))


class MixedPercisionActivationSearch8Bit(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [0, 0, 0, 0]

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg , self.expected_config)


class MixedPercisionActivationSearch2Bit(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [2, 8, 8, 2]

    def get_kpi(self):
        return KPI(96, 768)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg , self.expected_config)


class MixedPercisionActivationSearch4Bit(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 4, 4, 1]

    def get_kpi(self):
        return KPI(192, 1536)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg , self.expected_config)


class MixedPercisionActivationSearch4BitFunctional(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 4, 4, 1]

    def get_kpi(self):
        return KPI(81, 1536)

    def create_feature_network(self, input_shape):
        return MixedPrecisionFunctionalNet(input_shape)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg , self.expected_config)


class MixedPrecisionNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=5)
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.conv2(x)
        output = self.relu(x)
        return output


class MixedPrecisionFunctionalNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionFunctionalNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels, 3, kernel_size=3)

    def forward(self, inp):
        x1 = self.conv1(inp)
        x2 = self.conv2(inp)
        output = x1 + x2
        return output
