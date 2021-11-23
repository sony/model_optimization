# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.engine.sequential import Sequential
from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class NestedModelUnusedInputsOutputsTest(BaseFeatureNetworkTest):
    """
    Test two inner models when one's some outputs are not connected to any layer.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=5)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE, mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, True, True, True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 64, 64, 3]]

    def inner_functional_model(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 4)(inputs)
        y = layers.Conv2D(3, 4)(inputs)
        z = layers.Conv2D(3, 4)(inputs)
        w = layers.Conv2D(3, 4)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Activation('swish')(x)
        return keras.Model(inputs=inputs, outputs=[outputs, y, z, w])

    def second_inner_functional_model(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        inputs2 = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 4)(inputs)
        y = layers.Conv2D(3, 4)(inputs2)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, y])
        outputs = layers.Activation('swish')(x)
        return keras.Model(inputs=[inputs, inputs2], outputs=outputs)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(3, 4)(inputs)
        x = layers.Conv2D(3, 4)(x)
        x, y, z, w = self.inner_functional_model(x.shape)(x)
        x = self.second_inner_functional_model(x.shape)([z, y])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        for l in quantized_model.layers:
            if hasattr(l, 'layer'):
                self.unit_test.assertFalse(isinstance(l.layer, Functional) or isinstance(l.layer, Sequential))
            else:
                self.unit_test.assertFalse(isinstance(l, Functional) or isinstance(l, Sequential))
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
