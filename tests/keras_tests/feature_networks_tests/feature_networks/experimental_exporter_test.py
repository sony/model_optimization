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


import tensorflow as tf
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.activation_quantize_config import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_activation_quantize_config \
    import \
    WeightsActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_quantize_config import \
    WeightsQuantizeConfig
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class ExperimentalExporterTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4)(inputs)
        x = layers.Conv2D(3, 4)(x)
        outputs = layers.ReLU()(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[0], InputLayer))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], QuantizeWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1].layer, InputLayer))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1].quantize_config, ActivationQuantizeConfig))

        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], QuantizeWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].layer, layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].quantize_config, WeightsActivationQuantizeConfig))

        self.unit_test.assertTrue(isinstance(quantized_model.layers[3], QuantizeWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].layer, layers.Conv2D))
        self.unit_test.assertTrue(
            isinstance(quantized_model.layers[3].quantize_config, WeightsQuantizeConfig))

        self.unit_test.assertTrue(isinstance(quantized_model.layers[4], QuantizeWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[4].layer, layers.ReLU))
        self.unit_test.assertTrue(
            isinstance(quantized_model.layers[4].quantize_config, ActivationQuantizeConfig))

