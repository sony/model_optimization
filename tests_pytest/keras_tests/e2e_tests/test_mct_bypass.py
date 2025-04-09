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
import pytest

from model_compression_toolkit.gptq import keras_gradient_post_training_quantization
from model_compression_toolkit.ptq import keras_post_training_quantization

import keras
import numpy as np
from tests_pytest._fw_tests_common_base.base_mct_bypass_test import BaseMCTBypassTest


class TestKerasMCTBypass(BaseMCTBypassTest):

    def _build_test_model(self):
        x = keras.layers.Input((3, 3, 8))
        y = keras.layers.Conv2D(filters=8, kernel_size=3)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.ReLU()(y)
        return keras.Model(inputs=x, outputs=y)

    def _assert_models_equal(self, model, out_model):
        weights_model = model.get_weights()
        weights_out_model = out_model.get_weights()
        assert len(weights_model) == len(weights_out_model), "Different number of weight tensors between input and output models."
        for i, (w1, w2) in enumerate(zip(weights_model, weights_out_model)):
            assert np.array_equal(w1, w2), f"Mismatch in weight tensor at index {i}."

    @pytest.mark.parametrize('api_func', [keras_post_training_quantization,
                                          keras_gradient_post_training_quantization])
    def test_post_training_quantization_bypass(self, api_func):
        """This test is designed to verify that a Keras model, when processed through MCT API with a bypass flag
        enabled, retains its original architecture and parameters"""
        self._test_mct_bypass(api_func)
