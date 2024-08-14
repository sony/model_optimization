# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np
import tensorflow as tf
if tf.__version__ >= "2.13":
    from keras.src.layers import Conv2D, Multiply, Activation
    from keras.src.activations import sigmoid
else:
    from keras.layers import Conv2D, Multiply, Activation
    from keras.activations import sigmoid

from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class SigMulSubstitutionTest(BaseKerasFeatureNetworkTest):

    def get_tpc(self):
        tp = generate_test_tp_model({'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="test_no_quant", tp_model=tp)

    def create_networks(self):
        _in = tf.keras.layers.Input(self.input_shape[1:])
        x = Conv2D(4, 3, padding='same')(_in)
        x = tf.multiply(x, tf.sigmoid(x))  # This should be substituted.
        x = Conv2D(5, 3, padding='same')(x)
        x = tf.multiply(sigmoid(x), x)  # This should be substituted.
        x = Conv2D(5, 3, padding='same')(x)
        x = Multiply()([tf.sigmoid(x), x])  # This should be substituted.
        x = Conv2D(5, 3, padding='same')(x)
        x = Multiply()([x, Activation('sigmoid')(x)])  # This should be substituted.
        x = Conv2D(5, 3, padding='same')(x)
        x = Multiply()([x+1, tf.sigmoid(x)])  # This should not be substituted.
        x = Conv2D(5, 3, padding='same')(x)
        s = tf.sigmoid(x)
        x = Multiply()([x, s])  # This should not be substituted because the sigmoid node is an output.
        x = Conv2D(5, 3, padding='same')(x)
        x = Multiply()([x, tf.sigmoid(x)])
        return tf.keras.Model(inputs=_in, outputs=[x, s])

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        out_float = float_model(input_x)[0]
        out_quant = quantized_model(input_x)[0]
        cs = cosine_similarity(out_float.numpy(), out_quant.numpy())
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')

        self.unit_test.assertTrue(len(get_layers_from_model_by_type(quantized_model, tf.nn.silu)) == 5,
                                  "Not all Sigmoid-Mul functions were substituted.")
