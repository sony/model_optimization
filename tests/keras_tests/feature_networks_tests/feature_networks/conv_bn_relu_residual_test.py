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


import tensorflow as tf

from mct_quantizers import KerasActivationQuantizationHolder
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class ConvBnReluResidualTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        y = layers.Conv2D(7, 8)(inputs)
        x = layers.BatchNormalization()(y)
        x = layers.Activation('relu')(x)
        outputs = layers.Add()([x, y])
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        holders = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        add_layer = get_layers_from_model_by_type(quantized_model, layers.Add)[0]
        bn_layer = get_layers_from_model_by_type(quantized_model, layers.BatchNormalization)[0]

        self.unit_test.assertTrue(holders[1].output.ref() in [t.ref() for t in add_layer.input])
        self.unit_test.assertTrue(holders[3].output.ref() in [t.ref() for t in add_layer.input])
        self.unit_test.assertTrue(isinstance(bn_layer, layers.BatchNormalization)) # assert not folding

