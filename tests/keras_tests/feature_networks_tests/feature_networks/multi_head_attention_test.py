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


from tests.keras_tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.common.user_info import UserInformation
import tensorflow as tf
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultiHeadAttentionTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, sequence_length, d_model_k, d_model_v, num_heads, key_dim, value_dim):
        super().__init__(unit_test)
        self.sequence_length = sequence_length
        self.d_model_k = d_model_k
        self.d_model_v = d_model_v
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

    def create_inputs_shape(self):
        return [[self.val_batch_size, self.sequence_length, self.d_model_k + self.d_model_v]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        # TODO: check when key and value are separated inputs
        outputs = layers.MultiHeadAttention(self.num_heads, self.key_dim, value_dim=self.value_dim,
                                            kernel_initializer="glorot_uniform",
                                            bias_initializer="glorot_uniform")(inputs[:, :, :self.d_model_k],
                                                                               inputs[:, :, self.d_model_k:])
        # outputs = layers.MultiHeadAttention(self.num_heads, self.key_dim, value_dim=self.value_dim)(outputs, outputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        kernel_op_layer_index = 2
        self.unit_test.assertTrue(isinstance(quantized_model.layers[kernel_op_layer_index], self.kernel_op_layer))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[kernel_op_layer_index + 1], (layers.ReLU,
                                                                                                 layers.Activation,
                                                                                                 layers.PReLU)))
