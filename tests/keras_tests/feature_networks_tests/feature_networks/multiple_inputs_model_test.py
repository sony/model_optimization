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
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultipleInputsModelTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_of_inputs=3)


    def create_networks(self):
        inputs_1 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_2 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_3 = layers.Input(shape=self.get_input_shapes()[0][1:])
        x1 = layers.Conv2D(3, 4)(inputs_1)
        x2 = layers.Conv2D(3, 4)(inputs_2)
        x3 = layers.Conv2D(3, 4)(inputs_3)
        outputs = layers.Concatenate()([x1, x2, x3])
        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.inputs) == 3)
