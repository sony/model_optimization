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


class NativeResizeBilinearTest(BaseKerasFeatureNetworkTest):
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
        c1 = tf.concat([inputs, inputs2], axis=0)
        c1 = tf.image.resize(c1, [10, 20], preserve_aspect_ratio=False)
        boxes = tf.random.uniform(shape=(5, 4))
        box_indices = tf.random.uniform(shape=(5,), minval=0,
                                        maxval=1, dtype=tf.int32)
        outputs = tf.image.crop_and_resize(c1, boxes, box_indices, (24, 24), extrapolation_value=0)
        return keras.Model(inputs=[inputs, inputs2], outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        num_oplambda_layers = len([x for x in quantized_model.layers if isinstance(x, TFOpLambda)])
        assert num_oplambda_layers == 8, f'Num OPLambda: {num_oplambda_layers}'
        self.unit_test.assertTrue(quantized_model.layers[6].function == tf.image.resize)
        self.unit_test.assertTrue(quantized_model.layers[6].outbound_nodes[0].layer.function == tf.quantization.fake_quant_with_min_max_vars)
        self.unit_test.assertTrue(quantized_model.layers[8].function == tf.image.crop_and_resize)
        self.unit_test.assertTrue(quantized_model.layers[8].outbound_nodes[0].layer.function == tf.quantization.fake_quant_with_min_max_vars)
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
