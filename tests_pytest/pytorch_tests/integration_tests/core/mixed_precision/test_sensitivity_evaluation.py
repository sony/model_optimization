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
import numpy as np
import pytest
from torch import nn

from model_compression_toolkit.core.pytorch import constants
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests_pytest._fw_tests_common_base.base_sensitivity_eval_integ_test import BaseSensitivityEvaluationIntegTester
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin


class Model(nn.Module):
    def __init__(self, input_chw):
        super().__init__()
        c, h, w = input_chw
        self.conv = nn.Conv2d(c, 4, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv_tr = nn.ConvTranspose2d(4, 8, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(h*w*8, 10)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.conv_tr(y)
        y = self.flatten(y)
        y = self.fc(y)
        return y


class TestSensitivityEvaluation(BaseSensitivityEvaluationIntegTester, TorchFwMixin):
    BIAS = constants.BIAS
    KERNEL = constants.KERNEL
    conv_cls = nn.Conv2d
    relu_cls = nn.ReLU
    convtr_cls = nn.ConvTranspose2d
    fc_cls = nn.Linear

    def build_model(self, input_shape):
        return Model(input_chw=input_shape[1:])

    def infer_models(self, orig_model, mp_model, x: np.ndarray):
        x = to_torch_tensor(x)
        y = orig_model.to(get_working_device())(x)
        y_mp = mp_model(x)[-1]
        return y.detach().cpu().numpy(), y_mp.detach().cpu().numpy()

    def test_build_models(self):
        super().test_build_models()

    def test_build_models_disable_activations(self):
        super().test_build_models_disable_activations()

    def test_configure_mp_model(self):
        super().test_configure_mp_model()

    def test_configure_mp_model_errors(self):
        super().test_configure_mp_model_errors()

    @pytest.mark.parametrize('custom', [False, True])
    def test_compute_metric_method(self, custom, mocker):
        super()._run_test_compute_metric_method(custom, mocker)
