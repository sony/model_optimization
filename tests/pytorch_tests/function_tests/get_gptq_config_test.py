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
from model_compression_toolkit import get_pytorch_gptq_config, pytorch_gradient_post_training_quantization_experimental, \
    CoreConfig, QuantizationConfig, QuantizationErrorMethod, RoundingType
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

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
    yield [np.random.random((1, 3, 8, 8))]


class TestGetGPTQConfig(BasePytorchTest):

    def __init__(self, unit_test, quantization_method=QuantizationMethod.SYMMETRIC, rounding_type=RoundingType.STE):
        super().__init__(unit_test)
        self.quantization_method = quantization_method
        self.rounding_type = rounding_type

    def run_test(self):
        qc = QuantizationConfig(QuantizationErrorMethod.MSE,
                                QuantizationErrorMethod.MSE,
                                weights_bias_correction=False)  # disable bias correction when working with GPTQ
        cc = CoreConfig(quantization_config=qc)

        gptqv2_configurations = [get_pytorch_gptq_config(n_epochs=1,
                                                         optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4))]
        for config in gptqv2_configurations:
            config.rounding_type = self.rounding_type

        tp = generate_test_tp_model({'weights_quantization_method': self.quantization_method})
        symmetric_weights_tpc = generate_pytorch_tpc(name="gptq_config_test", tp_model=tp)

        for i, gptq_config in enumerate(gptqv2_configurations):
            pytorch_gradient_post_training_quantization_experimental(model=TestModel(),
                                                                     representative_data_gen=random_datagen_experimental,
                                                                     core_config=cc,
                                                                     gptq_config=gptq_config,
                                                                     target_platform_capabilities=symmetric_weights_tpc)
