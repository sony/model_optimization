# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import nn

import model_compression_toolkit as mct
from model_compression_toolkit.gptq import get_pytorch_gptq_config, pytorch_gradient_post_training_quantization_experimental, RoundingType
from model_compression_toolkit import CoreConfig, QuantizationConfig, QuantizationErrorMethod, DefaultDict
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR, MAX_LSB_STR
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from model_compression_toolkit.core.pytorch.utils import get_working_device, set_model, to_torch_tensor, \
    torch_tensor_to_numpy

tp = mct.target_platform


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(4, 4), stride=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.prelu = torch.nn.PReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(4, 4), stride=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


def random_datagen_experimental():
    for _ in range(20):
        yield [np.random.random((1, 3, 8, 8))]


class TestGetGPTQConfig(BasePytorchTest):

    def __init__(self, unit_test, quantization_method=QuantizationMethod.SYMMETRIC, rounding_type=RoundingType.STE,
                 train_bias=False, quantization_parameters_learning=False):
        super().__init__(unit_test)
        self.quantization_method = quantization_method
        self.rounding_type = rounding_type
        self.train_bias = train_bias
        self.quantization_parameters_learning = quantization_parameters_learning

    def run_test(self):
        qc = QuantizationConfig(QuantizationErrorMethod.MSE,
                                QuantizationErrorMethod.MSE,
                                weights_bias_correction=False)  # disable bias correction when working with GPTQ
        cc = CoreConfig(quantization_config=qc)

        gptqv2_config = get_pytorch_gptq_config(n_epochs=1,
                                                optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4))
        gptqv2_config.rounding_type = self.rounding_type
        gptqv2_config.train_bias = self.train_bias

        if self.rounding_type == RoundingType.SoftQuantizer:
            gptqv2_config.gptq_quantizer_params_override = \
                {QUANT_PARAM_LEARNING_STR: self.quantization_parameters_learning}
        elif self.rounding_type == RoundingType.STE:
            gptqv2_config.gptq_quantizer_params_override = \
                {MAX_LSB_STR: DefaultDict({}, lambda: 1)}
        else:
            gptqv2_config.gptq_quantizer_params_override = None

        tp = generate_test_tp_model({'weights_quantization_method': self.quantization_method})
        symmetric_weights_tpc = generate_pytorch_tpc(name="gptq_config_test", tp_model=tp)

        float_model = TestModel()

        quant_model, _ = pytorch_gradient_post_training_quantization_experimental(model=float_model,
                                                                                  representative_data_gen=random_datagen_experimental,
                                                                                  core_config=cc,
                                                                                  gptq_config=gptqv2_config,
                                                                                  target_platform_capabilities=symmetric_weights_tpc)
