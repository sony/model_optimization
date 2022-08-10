# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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


import tensorflow as tf
import numpy as np

from model_compression_toolkit.core.common.network_editors import EditRule, NodeTypeFilter
from model_compression_toolkit.core.common.network_editors.actions import ReplaceLayer
from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers


def get_new_weights_for_identity_dw_conv2d_layer(weights={}, **kwargs):
    """
    return the weights of depthwise conv2d layers set to ones
    """

    new_weights = weights
    key = list(weights.keys())[0]
    old_kernel_shape = weights[key].shape
    new_kernel = np.ones(old_kernel_shape)
    new_weights[key] = new_kernel
    return new_weights, kwargs


class DwConv2dReplacementTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return get_quantization_disabled_keras_tpc("depthwise_conv2d_replacement_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = layers.DepthwiseConv2D(1, use_bias=False)(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(np.isclose(0, np.mean(quantized_model.predict(input_x) - input_x)))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], layers.DepthwiseConv2D))
        self.unit_test.assertTrue(np.all(quantized_model.layers[1].depthwise_kernel.numpy() == 1))

    def get_network_editor(self):
        return [EditRule(filter=NodeTypeFilter(layers.DepthwiseConv2D),
                         action=ReplaceLayer(layers.DepthwiseConv2D, get_new_weights_for_identity_dw_conv2d_layer))
                ]


