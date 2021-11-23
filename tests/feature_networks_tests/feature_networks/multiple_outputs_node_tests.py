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


from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultipleOutputsNodeTests(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING, mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, True, False, True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU(max_value=6.0)(x)
        outputs = layers.Dense(20)(x)
        outputs = tf.split(outputs, num_or_size_splits=20, axis=-1)
        return keras.Model(inputs=inputs, outputs=[outputs[0], outputs[4], outputs[2]])

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.outputs) == 3)
        inputs = self.generate_inputs(self.create_inputs_shape())
        output_q = quantized_model.predict(inputs)
        output_f = float_model.predict(inputs)
        for o_q, o_f in zip(output_q, output_f):
            cs = cosine_similarity(o_f, o_q)
            self.unit_test.assertTrue(np.isclose(cs, 1))
