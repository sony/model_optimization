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
from functools import partial
import torch
import torch.nn as nn
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.constants import PYTORCH
from mct_quantizers import PytorchQuantizationWrapper

tp = mct.target_platform


class ConstQuantizationNet(nn.Module):
    def __init__(self, layer, const):
        super().__init__()
        self.layer = layer
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const

    def forward(self, x):
        return self.layer(x, self.const)


class ConstQuantizationReverseOrderNet(nn.Module):
    def __init__(self, layer, const):
        super().__init__()
        self.layer = layer
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const

    def forward(self, x):
        return self.layer(self.const, x)


class ConstQuantizationTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test, func, const, input_reverse_order=False):
        super().__init__(unit_test=unit_test, input_shape=(16, 32, 32))
        self.func = func
        self.const = const
        self.input_reverse_order = input_reverse_order

    def generate_inputs(self):
        return [np.random.random(in_shape)+1 for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        return mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, "v3")

    def create_networks(self):
        if self.input_reverse_order:
            return ConstQuantizationReverseOrderNet(self.func, self.const)
        else:
            return ConstQuantizationNet(self.func, self.const)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        set_model(float_model)
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(torch_tensor_to_numpy(y), torch_tensor_to_numpy(y_hat))
        self.unit_test.assertTrue(np.isclose(cs, 1, atol=0.001), msg=f'fail cosine similarity check: {cs}')
        for n, m in quantized_model.named_modules():
            if n == self.func.__name__:
                self.unit_test.assertTrue(isinstance(m, PytorchQuantizationWrapper),
                                          msg=f'Expected layer type to be "PytorchQuantizationWrapper" but got {type(m)}.')
                self.unit_test.assertTrue((list(m.weight_values.values())[0].detach().cpu().numpy() ==
                                           self.const).all(),
                                          msg=f'Expected PytorchQuantizationWrapper const value to match float const.')


class AdvancedConstQuantizationNet(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 130, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(130, 16, 3)
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.add(x, self.const)
        x = self.conv2(x)
        return x


class AdvancedConstQuantizationTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test, input_shape=(3, 32, 32), num_calibration_iter=32)
        self.const = (np.random.random((130, 1, 1))).astype(np.float32)

    def get_ptq_facade(self):
        gptq_config = mct.gptq.get_pytorch_gptq_config(30)
        return partial(mct.gptq.pytorch_gradient_post_training_quantization,
                       gptq_config=gptq_config)

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(9e3)

    def generate_inputs(self):
        return [np.random.random(in_shape)+1 for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        return mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, "v3")

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig()

    def create_networks(self):
        return AdvancedConstQuantizationNet(self.const)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        set_model(float_model)
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        for n, m in quantized_model.named_modules():
            if n == torch.add.__name__:
                self.unit_test.assertTrue(isinstance(m, PytorchQuantizationWrapper),
                                          msg=f'Expected layer type to be "PytorchQuantizationWrapper" but got {type(m)}.')
                self.unit_test.assertTrue((list(m.weight_values.values())[0].detach().cpu().numpy() ==
                                           self.const).all(),
                                          msg=f'Expected PytorchQuantizationWrapper const value to match float const.')
