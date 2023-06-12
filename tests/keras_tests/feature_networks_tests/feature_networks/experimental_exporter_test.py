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
from mct_quantizers import KerasQuantizationWrapper, BaseInferableQuantizer, KerasActivationQuantizationHolder

from model_compression_toolkit.core.keras.constants import KERNEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class ExportableModelTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, experimental_exporter=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4)(inputs)
        x = layers.Conv2D(3, 4)(x)
        outputs = layers.ReLU()(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        holders_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)

        self.unit_test.assertTrue(len(holders_layer) == 3)
        for holder in holders_layer:
            self.unit_test.assertTrue(isinstance(holder.activation_holder_quantizer, BaseInferableQuantizer))

        self.unit_test.assertTrue(len(conv_layers[0].weights_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(conv_layers[0].weights_quantizers[KERNEL], BaseInferableQuantizer))

        self.unit_test.assertTrue(len(conv_layers[1].weights_quantizers) == 1)
        self.unit_test.assertTrue(isinstance(conv_layers[1].weights_quantizers[KERNEL], BaseInferableQuantizer))
