# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
import model_compression_toolkit as mct
import tensorflow as tf

from tests.keras_tests.tpc_keras import get_16bit_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import is_layer_fake_quant

keras = tf.keras
layers = keras.layers


class BaseInputScalingTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return get_16bit_tpc("input_scaling_range_test")

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                      mct.QuantizationErrorMethod.NOCLIPPING,
                                      input_scaling=True)


    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        qi = 3 if isinstance(quantized_model.layers[2], layers.ZeroPadding2D) else 2
        fi = 2 if isinstance(float_model.layers[1], layers.ZeroPadding2D) else 1
        self.unit_test.assertTrue(is_layer_fake_quant(quantized_model.layers[1]))
        self.unit_test.assertTrue(quantization_info.input_scale != 1)
        alpha = (float_model.layers[fi].weights[0] / quantized_model.layers[qi].weights[0]).numpy().mean()
        self.unit_test.assertTrue(np.allclose(alpha, quantization_info.input_scale, atol=1e-1))



class InputScalingDenseTest(BaseInputScalingTest):
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class InputScalingConvTest(BaseInputScalingTest):
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class InputScalingDWTest(BaseInputScalingTest):
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same')(inputs)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class InputScalingZeroPadTest(BaseInputScalingTest):
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.ZeroPadding2D()(inputs)
        x = layers.DepthwiseConv2D(1, padding='same')(x)
        x = layers.ReLU()(x)
        outputs = layers.Dense(30)(x)
        return keras.Model(inputs=inputs, outputs=outputs)
