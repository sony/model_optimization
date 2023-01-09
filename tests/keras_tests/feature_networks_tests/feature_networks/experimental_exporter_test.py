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
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer

from model_compression_toolkit.core.keras.constants import KERNEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers
from model_compression_toolkit import quantizers_infrastructure as qi

class ExportableModelTest(BaseKerasFeatureNetworkTest):
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
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], qi.KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1].layer, Layer))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1].dispatcher, qi.KerasNodeQuantizationDispatcher))
        self.unit_test.assertTrue(len(quantized_model.layers[1].dispatcher.activation_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1].dispatcher.activation_quantizers[0], qi.BaseInferableQuantizer))

        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], qi.KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].layer, layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].dispatcher, qi.KerasNodeQuantizationDispatcher))
        self.unit_test.assertTrue(len(quantized_model.layers[2].dispatcher.activation_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].dispatcher.activation_quantizers[0], qi.BaseInferableQuantizer))
        self.unit_test.assertTrue(len(quantized_model.layers[2].dispatcher.weight_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].dispatcher.weight_quantizers[KERNEL], qi.BaseInferableQuantizer))

        self.unit_test.assertTrue(isinstance(quantized_model.layers[3], qi.KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].layer, layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].dispatcher, qi.KerasNodeQuantizationDispatcher))
        self.unit_test.assertTrue(len(quantized_model.layers[3].dispatcher.activation_quantizers) == 0)
        self.unit_test.assertTrue(len(quantized_model.layers[3].dispatcher.weight_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].dispatcher.weight_quantizers[KERNEL], qi.BaseInferableQuantizer))

        self.unit_test.assertTrue(isinstance(quantized_model.layers[4], qi.KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[4].layer, layers.ReLU))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[4].dispatcher, qi.KerasNodeQuantizationDispatcher))
        self.unit_test.assertTrue(len(quantized_model.layers[4].dispatcher.activation_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[4].dispatcher.activation_quantizers[0], qi.BaseInferableQuantizer))

