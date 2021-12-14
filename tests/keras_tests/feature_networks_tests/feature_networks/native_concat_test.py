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

from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class NativeConcateTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=1)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                      mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO,
                                      mct.QuantizationMethod.POWER_OF_TWO,
                                      16,
                                      16,
                                      False,
                                      False,
                                      True)

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, 3], [self.val_batch_size, 16, 16, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs2 = layers.Input(shape=self.get_input_shapes()[1][1:])
        outputs = tf.concat([inputs, inputs2], axis=0)
        return keras.Model(inputs=[inputs, inputs2], outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        num_oplambda_layers = len([x for x in quantized_model.layers if isinstance(x, TFOpLambda)])
        self.unit_test.assertTrue(num_oplambda_layers == 4)
        self.unit_test.assertTrue(quantized_model.layers[4].function == tf.concat)
        self.unit_test.assertTrue(quantized_model.layers[4].outbound_nodes[0].layer.function == tf.quantization.fake_quant_with_min_max_vars)
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
