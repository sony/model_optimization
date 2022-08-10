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
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
import model_compression_toolkit as mct
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.common.user_info import UserInformation
import tensorflow as tf
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultiHeadAttentionTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, input_shapes, num_heads, query_key_dim, value_dim,
                 attention_axes=None, separate_key_value=False, output_dim=None):
        super().__init__(unit_test)
        self.num_calibration_iter = 100

        query_input_shape, key_input_shape, value_input_shape = input_shapes
        self.query_input_shape = query_input_shape
        self.key_input_shape = key_input_shape
        self.value_input_shape = value_input_shape
        self.num_heads = num_heads
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.attention_axes=attention_axes
        self.separate_key_value = separate_key_value
        self.output_dim = output_dim

    def get_tpc(self):
        return get_quantization_disabled_keras_tpc("multi_head_attention_test")

    def get_input_shapes(self):
        if self.separate_key_value:
            return [[self.val_batch_size] + list(self.query_input_shape),
                    [self.val_batch_size] + list(self.key_input_shape),
                    [self.val_batch_size] + list(self.value_input_shape)]
        else:
            return [[self.val_batch_size] + list(self.query_input_shape),
                    [self.val_batch_size] + list(self.value_input_shape)]

    def create_networks(self):
        if self.separate_key_value:
            query_input_shape, key_input_shape, value_input_shape = self.get_input_shapes()
        else:
            query_input_shape, value_input_shape = self.get_input_shapes()
            key_input_shape = None
        q_input = layers.Input(shape=query_input_shape[1:])
        k_input = layers.Input(shape=key_input_shape[1:]) if self.separate_key_value else None
        v_input = layers.Input(shape=value_input_shape[1:])

        mha_layer = layers.MultiHeadAttention(self.num_heads, self.query_key_dim, value_dim=self.value_dim,
                                              attention_axes=self.attention_axes, output_shape=self.output_dim,
                                              kernel_initializer="glorot_uniform",
                                              bias_initializer="glorot_uniform",
                                              )
        outputs = mha_layer(q_input, v_input, key=k_input)
        inputs_list = [q_input, k_input, v_input] if self.separate_key_value else [q_input, v_input]
        return keras.Model(inputs=inputs_list, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        out_quantized = quantized_model(input_x).numpy()
        out_float = float_model(input_x).numpy()
        self.unit_test.assertTrue(out_quantized.shape == out_float.shape)
        nmse = np.mean(np.abs((out_quantized - out_float)) ** 2) / np.mean(np.abs(out_float) ** 2)
        self.unit_test.assertTrue(np.isclose(nmse, 0))
