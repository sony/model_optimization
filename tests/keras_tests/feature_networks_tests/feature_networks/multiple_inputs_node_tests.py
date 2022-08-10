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
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf

from tests.keras_tests.tpc_keras import get_16bit_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultipleInputsNodeTests(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return get_16bit_tpc("multiple_inputs_test")

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING,
                                      True, False, True)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU(max_value=6.0)(x)
        outputs = layers.Dense(20)(x)
        outputs = layers.Add()([outputs, x])
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        inputs = self.generate_inputs()
        output_q = quantized_model.predict(inputs)
        output_f = float_model.predict(inputs)
        cs = cosine_similarity(output_f, output_q)
        self.unit_test.assertTrue(np.isclose(cs, 1))
