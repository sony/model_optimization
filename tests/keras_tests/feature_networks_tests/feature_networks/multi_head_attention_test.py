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
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig,\
    ThresholdSelectionMethod
import model_compression_toolkit as mct
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.common.user_info import UserInformation
import tensorflow as tf
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultiHeadAttentionTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, q_sequence_length, kv_sequence_length, d_model_q, d_model_k, d_model_v,
                 num_heads, query_key_dim, value_dim, separate_key_value=False):
        super().__init__(unit_test)
        self.q_sequence_length = q_sequence_length
        self.kv_sequence_length = kv_sequence_length
        self.separate_key_value = separate_key_value
        self.d_model_q = d_model_q
        self.d_model_k = 0 if separate_key_value else d_model_k
        self.d_model_v = d_model_v
        self.num_heads = num_heads
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim

    def get_quantization_config(self):
        return QuantizationConfig(activation_threshold_method=ThresholdSelectionMethod.NOCLIPPING,
                                  weights_threshold_method=ThresholdSelectionMethod.NOCLIPPING,
                                  activation_n_bits=16, weights_n_bits=16,
                                  )

    def create_inputs_shape(self):
        return [[self.val_batch_size, self.q_sequence_length + self.kv_sequence_length,
                 self.d_model_q + self.d_model_k + self.d_model_v]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        mha_layer = layers.MultiHeadAttention(self.num_heads, self.query_key_dim, value_dim=self.value_dim,
                                              kernel_initializer="glorot_uniform",
                                              bias_initializer="glorot_uniform")
        if self.separate_key_value:
            outputs = mha_layer(inputs[:, :self.q_sequence_length, :self.d_model_q],
                                inputs[:, self.q_sequence_length:, self.d_model_q:self.d_model_q + self.d_model_v])
        else:
            outputs = mha_layer(inputs[:, :self.q_sequence_length, :self.d_model_q],
                                inputs[:, self.q_sequence_length:, self.d_model_q:self.d_model_q+self.d_model_v],
                                key=inputs[:, self.q_sequence_length:, self.d_model_q+self.d_model_v:])
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        out_quantized = quantized_model(input_x[0]).numpy().flatten()
        out_float = float_model(input_x[0]).numpy().flatten()
        nmse = np.mean(np.abs((out_quantized - out_float)) ** 2) / np.mean(np.abs(out_float) ** 2)
        self.unit_test.assertTrue(np.isclose(nmse, 0))
