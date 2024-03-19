# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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


from packaging import version
import tensorflow as tf
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
import model_compression_toolkit as mct

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity


class MatmulToDenseSubstitutionTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, input_shape=(8,))

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 16,
                                     'activation_n_bits': 16,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="no_quantization", tp_model=tp)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                           mct.core.QuantizationErrorMethod.NOCLIPPING, False, False)

    def create_networks(self):
        inputs = tf.keras.layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.matmul(inputs, b=tf.random.normal((8, 10)))
        x = tf.keras.layers.ReLU()(x)
        x = tf.matmul(x, np.random.normal(size=(10, 16)))
        x = tf.keras.layers.ReLU()(x)
        x = tf.matmul(x, np.random.normal(size=(16, 32)).tolist())
        x = tf.matmul(tf.reshape(x, (-1, 8, 4)),
                      tf.reshape(x, (-1, 4, 8)))
        x = tf.keras.layers.ReLU()(tf.reshape(x, (-1, 64)))
        x = tf.matmul(x, tf.random.normal((11, 64)), transpose_b=True)
        x = tf.keras.layers.ReLU()(x)
        x = tf.matmul(x, tf.random.normal((10, 11)), False, True)
        x = tf.keras.layers.ReLU()(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check the output didn't change
        y = float_model(input_x).numpy()
        y_hat = quantized_model(input_x).numpy()
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')

        num_matmuls = 0
        for layer in quantized_model.layers:
            if isinstance(layer, TFOpLambda) and layer.symbol is TFOpLambda(tf.matmul).symbol:
                num_matmuls += 1

        # check all "matmul"s were replaced except the one with 2 tensor inputs
        self.unit_test.assertTrue(num_matmuls == 1, msg=f'Only one matmul should remain in the quantized model')
