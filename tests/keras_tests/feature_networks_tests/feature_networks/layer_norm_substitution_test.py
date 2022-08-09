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
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.tpc_keras import get_16bit_tpc
import tensorflow as tf
import numpy as np

keras = tf.keras
layers = keras.layers


class LayerNormSub(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, scale=True, center=True):
        super().__init__(unit_test)
        self.num_calibration_iter = 100
        self.scale = scale
        self.center = center

    def get_tpc(self):
        return get_16bit_tpc("layer_norm_sub")

    def create_networks(self):
        _input = layers.Input(shape=self.input_shape[1:])
        outputs = layers.LayerNormalization(center=self.center, scale=self.scale,
                                            beta_initializer="glorot_uniform",
                                            gamma_initializer="glorot_uniform")(_input)
        return keras.Model(inputs=_input, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        if self.center is not False or self.scale is not False:
            self.unit_test.assertTrue(isinstance(quantized_model.layers[4], layers.BatchNormalization))
        out_quantized = quantized_model(input_x).numpy()
        out_float = float_model(input_x).numpy()
        self.unit_test.assertTrue(out_quantized.shape == out_float.shape)
        nmse = np.mean(np.abs((out_quantized - out_float)) ** 2) / np.mean(np.abs(out_float) ** 2)
        self.unit_test.assertTrue(np.isclose(nmse, 0, atol=1e-7))
