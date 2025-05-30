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

from model_compression_toolkit.core.keras import constants
from tests_pytest._fw_tests_common_base.base_sensitivity_eval_integ_test import BaseSensitivityEvaluationIntegTester
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin

import keras


class TestSensitivityEvaluation(BaseSensitivityEvaluationIntegTester, KerasFwMixin):
    BIAS = constants.BIAS
    KERNEL = constants.KERNEL
    conv_cls = keras.layers.Conv2D
    relu_cls = keras.layers.ReLU
    convtr_cls = keras.layers.Conv2DTranspose
    fc_cls = keras.layers.Dense

    input_shape = (1, 16, 16, 3)

    def build_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Input(self.input_shape[1:]),
            keras.layers.Conv2D(4, kernel_size=3),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(8, kernel_size=3),
            keras.layers.Flatten(),
            keras.layers.Dense(10)
        ])
        return model

    def infer_models(self, orig_model, mp_model, x: np.ndarray):
        y = orig_model(x)
        y_mp = mp_model(x)[-1]
        return y.numpy(), y_mp.numpy()

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

