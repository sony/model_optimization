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
    from keras.src.layers.core import TFOpLambda
    from keras.src.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D
else:
    from keras.layers.core import TFOpLambda
    from keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D

import model_compression_toolkit as mct
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


# get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class ConvFuncSubstitutionsTest(BaseKerasFeatureNetworkTest):

    def get_tpc(self):
        tp = generate_test_tp_model({'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="test_no_quant", tp_model=tp)

    def create_networks(self):
        _in = tf.keras.layers.Input(self.input_shape[1:])
        x = ZeroPadding2D(padding=10)(_in)
        x = tf.nn.conv2d(x, np.random.random((3, 3, self.input_shape[-1], 2)).astype(np.float32),
                         [1, 1], padding='VALID')
        x = tf.add(x, np.random.random((2,)).astype(np.float32))
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, np.random.random((3, 3, 2, 2)).astype(np.float32),
                         [1, 1], padding='SAME', dilations=2)
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, np.random.random((3, 3, 2, 2)).astype(np.float32),
                         [1, 1], padding='SAME', dilations=[1, 2, 2, 1])
        x = tf.nn.relu(x)
        x = tf.nn.depthwise_conv2d(x, np.random.random((3, 3, 2, 1)).astype(np.float32),
                                   strides=(1, 2, 2, 1), padding='SAME')
        x = tf.add(x, np.random.random((2,)).astype(np.float32))
        x = tf.nn.relu(x)
        x = tf.nn.depthwise_conv2d(x, np.random.random((3, 3, 2, 1)).astype(np.float32),
                                   strides=(1, 1, 1, 1), padding='VALID', dilations=[2, 2])
        x = tf.nn.relu(x)
        x = tf.nn.convolution(x, np.random.random((3, 3, 2, 4)).astype(np.float32),
                              [2, 1], padding='SAME')
        x = tf.nn.bias_add(x, np.random.random((4,)).astype(np.float32))
        return tf.keras.Model(inputs=_in, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        out_float = float_model(input_x)
        out_quant = quantized_model(input_x)
        cs = cosine_similarity(out_float.numpy(), out_quant.numpy())
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')

        self.unit_test.assertTrue(len(get_layers_from_model_by_type(quantized_model, Conv2D)) == 4,
                                  "Not all conv functions were substituted.")
        self.unit_test.assertTrue(len(get_layers_from_model_by_type(quantized_model, DepthwiseConv2D)) == 2,
                                  "Not all dw-conv functions were substituted.")
        self.unit_test.assertTrue(len(get_layers_from_model_by_type(quantized_model, tf.add)) == 0,
                                  "Not all tf.add functions were absorbed as biases.")
        self.unit_test.assertTrue(len(get_layers_from_model_by_type(quantized_model, tf.nn.bias_add)) == 0,
                                  "Not all tf.add functions were absorbed as biases.")
