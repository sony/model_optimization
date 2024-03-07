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

from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.core import DebugConfig
from model_compression_toolkit.core.common.network_editors.actions import EditRule, ChangeFinalWeightsQuantConfigAttr, \
    ChangeFinalActivationQuantConfigAttr, ChangeCandidatesActivationQuantConfigAttr
from model_compression_toolkit.core.common.network_editors.node_filters import NodeTypeFilter
from model_compression_toolkit.core.keras.constants import KERNEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class ChangeFinalWeightQCAttrTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_debug_config(self):
        return DebugConfig(network_editor=[EditRule(filter=NodeTypeFilter(layers.Conv2D),
                                                    action=ChangeFinalWeightsQuantConfigAttr(weights_bias_correction=False))])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4, use_bias=False)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layer = get_layers_from_model_by_type(quantized_model, layers.Conv2D)[0]
        self.unit_test.assertTrue(conv_layer.layer.bias is None)  # If bias correction is enabled, a bias should be added -
        # This asserts the editing occured


class ChangeFinalActivationQCAttrTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test )

    def get_debug_config(self):
        return DebugConfig(network_editor=[EditRule(filter=NodeTypeFilter(layers.Conv2D),
                                                    action=ChangeFinalActivationQuantConfigAttr(activation_n_bits=7))])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4, use_bias=False)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1]
        self.unit_test.assertTrue(conv_holder_layer.activation_holder_quantizer.get_config()['num_bits'] == 7)
