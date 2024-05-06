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

import itertools
import unittest

import numpy as np
import torch.nn

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
import torch

class ModelToTest(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.bn = torch.nn.BatchNorm2d(num_features=3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def model_gen():
    return ModelToTest()


class TestQuantizationConfigurations(unittest.TestCase):
    def test_run_quantization_config(self):
        x = np.random.randn(1, 3, 16, 16)

        def representative_data_gen():
            yield [x]

        quantizer_methods = [mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                             mct.target_platform.QuantizationMethod.SYMMETRIC,
                             mct.target_platform.QuantizationMethod.UNIFORM]

        quantization_error_methods = [mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.NOCLIPPING,
                                      mct.core.QuantizationErrorMethod.MAE,
                                      mct.core.QuantizationErrorMethod.LP]
        bias_correction = [True, False]
        relu_bound_to_power_of_2 = [True, False]
        weights_per_channel = [True, False]
        shift_negative_correction = [True, False]

        weights_config_list = [quantizer_methods, quantization_error_methods, bias_correction, weights_per_channel]
        weights_test_combinations = list(itertools.product(*weights_config_list))

        activation_config_list = [quantizer_methods, quantization_error_methods, relu_bound_to_power_of_2,
                                  shift_negative_correction]
        activation_test_combinations = list(itertools.product(*activation_config_list))

        model = model_gen()
        for quantize_method, error_method, bias_correction, per_channel in weights_test_combinations:
            tp = generate_test_tp_model({
                'weights_quantization_method': quantize_method,
                'weights_n_bits': 8,
                'activation_n_bits': 16,
                'weights_per_channel_threshold': per_channel})
            tpc = generate_pytorch_tpc(name="quant_config_weights_test", tp_model=tp)

            qc = mct.core.QuantizationConfig(activation_error_method=mct.core.QuantizationErrorMethod.NOCLIPPING,
                                             weights_error_method=error_method,
                                             relu_bound_to_power_of_2=False,
                                             weights_bias_correction=bias_correction)
            core_config = mct.core.CoreConfig(quantization_config=qc)
            _, _ = mct.ptq.pytorch_post_training_quantization(model,
                                                              representative_data_gen,
                                                              core_config=core_config,
                                                              target_platform_capabilities=tpc)

        model = model_gen()
        for quantize_method, error_method, relu_bound_to_power_of_2, shift_negative_correction in activation_test_combinations:
            tp = generate_test_tp_model({
                'activation_quantization_method': quantize_method,
                'weights_n_bits': 16,
                'activation_n_bits': 8})
            tpc = generate_pytorch_tpc(name="quant_config_activation_test", tp_model=tp)

            qc = mct.core.QuantizationConfig(activation_error_method=error_method,
                                             weights_error_method=mct.core.QuantizationErrorMethod.NOCLIPPING,
                                             relu_bound_to_power_of_2=relu_bound_to_power_of_2,
                                             weights_bias_correction=False,
                                             shift_negative_activation_correction=shift_negative_correction)
            core_config = mct.core.CoreConfig(quantization_config=qc)

            _, _ = mct.ptq.pytorch_post_training_quantization(model,
                                                              representative_data_gen,
                                                              core_config=core_config,
                                                              target_platform_capabilities=tpc)


if __name__ == '__main__':
    unittest.main()