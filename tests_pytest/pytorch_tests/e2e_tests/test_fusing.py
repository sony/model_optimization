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
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core.pytorch.resource_utilization_data_facade import pytorch_resource_utilization_data

from model_compression_toolkit.constants import PYTORCH

from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
import torch.nn as nn


from model_compression_toolkit.ptq import pytorch_post_training_quantization
from tests_pytest._fw_tests_common_base.base_fusing_test import BaseFusingTest
from mct_quantizers import PytorchActivationQuantizationHolder

from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin


class TestPytorchFusing(BaseFusingTest, TorchFwMixin):

    bhwc_input_shape = (1, 3, 18, 18)

    fw_ptq_facade = pytorch_post_training_quantization
    tpc = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)
    fw_ru_data_facade = pytorch_resource_utilization_data

    def _build_test_model_reuse(self, input_shape):
        class FusingTestReuse(nn.Module):
            def __init__(self, input_channels=input_shape[1]):
                super(FusingTestReuse, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv1(x)
                x = self.relu(x)
                return x

        return FusingTestReuse()

    def _build_test_model_ru_data_facade(self, input_shape):
        class FusingTestRUDataFacade(nn.Module):
            def __init__(self, input_channels=input_shape[1]):
                super(FusingTestRUDataFacade, self).__init__()
                self.conv = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=1)
                self.relu = nn.ReLU()

            def forward(self, y):
                x = self.conv(y)
                x = self.relu(x)
                x = x + y
                return x

        return FusingTestRUDataFacade()

    def _build_test_model_basic_fusing(self, input_shape):
        class BasicFusingTestModel(nn.Module):
            def __init__(self, input_channels=input_shape[1]):
                super(BasicFusingTestModel, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
                self.sigmoid = nn.Sigmoid()
                self.flatten = nn.Flatten()
                self.dense = nn.Linear(in_features=392, out_features=10)
                self.hswish = nn.Hardswish()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.sigmoid(x)
                x = self.flatten(x)
                x = self.dense(x)
                x = self.hswish(x)
                return x

        return BasicFusingTestModel()

    def _build_test_model_snc(self, input_shape):
        class SncModel(nn.Module):
            def __init__(self, input_channels=input_shape[1]):
                super(SncModel, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3, padding=1)
                self.activation = nn.SiLU()
                self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
                self.conv3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)

            def forward(self, x):
                y = self.conv1(x)
                y = self.activation(y)
                x = x+y
                x = self.conv2(x)
                x = self.activation(x)
                x = self.conv3(x)
                x = self.activation(x)
                return x

        return SncModel()
    def _get_expected_act_quant_holders(self):
        return ['x_activation_holder_quantizer',
                'relu_activation_holder_quantizer',
                'sigmoid_activation_holder_quantizer',
                'hswish_activation_holder_quantizer']


    def _get_expected_act_quant_holders_in_reuse_model(self):
        return ['x_activation_holder_quantizer',
                'relu_activation_holder_quantizer',
                'relu_1_activation_holder_quantizer']

    def _get_actual_act_quant_holders(self, qmodel):
        return [k for k, v in qmodel.named_modules() if isinstance(v, PytorchActivationQuantizationHolder)]

    def test_quantized_model_contains_only_expected_activation_quantizers(self):
        """
        Runs PTQ and checks that the activation quantizers are the activation quantizers that we expect.
        """
        super().test_quantized_model_contains_only_expected_activation_quantizers()

    def test_quantized_model_with_reuse_contains_only_expected_activation_quantizers(self):
        """
        Runs PTQ on a model with reuse layer and checks that the activation quantizers are the activation quantizers that we expect.
        """
        super().test_quantized_model_with_reuse_contains_only_expected_activation_quantizers()

    def test_facade_ru_data_matches_expected_for_fused_graph(self):
        """
        Compute RU data on a model and check the computed max cut is as expected when we take the fusing into account.
        """
        super().test_facade_ru_data_matches_expected_for_fused_graph()

    def test_final_ru_data_is_correct(self):
        """
        Check that the activation memory in the final RU after running PTQ is as expected when we take fusing into account.
        """
        super().test_final_ru_data_is_correct()

    def test_facade_ru_data_matches_expected_with_snc_model(self):
        """
        Compute RU data on a model that goes through SNC and check the computed max cut is as expected when we take the fusing into account.
        """
        super().test_facade_ru_data_matches_expected_with_snc_model()

    def test_final_ru_data_with_snc_model(self):
        """
        Check that the activation memory in the final RU after running PTQ on a model that goes through SNC is as expected when we take fusing into account.
        """
        super().test_final_ru_data_with_snc_model()
