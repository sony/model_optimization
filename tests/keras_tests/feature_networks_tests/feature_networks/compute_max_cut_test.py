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

import model_compression_toolkit as mct
import tensorflow as tf

from mct_quantizers.keras.metadata import get_metadata
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class ComputeMaxCutTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:], name='input_layer')
        x = layers.Conv2D(4, 3, padding='same', name='conv2d_1')(inputs)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        y = layers.ReLU(name='relu_1')(x)
        x = layers.Conv2D(4, 3, padding='same', name='conv2d_2')(y)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.ReLU(name='relu_2')(x)
        outputs = layers.Add(name='add_layer')([x, y])
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_tpc(self):
        return mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v2")

    def get_debug_config(self):
        return mct.core.DebugConfig(simulate_scheduler=True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        _metadata = get_metadata(quantized_model)
        self.unit_test.assertEqual(_metadata['scheduling_info']['operators_scheduling'],
                                   ['InputLayer:input_layer',
                                    'FusedLayerType:FusedNode_conv2d_1_bn_relu_1',
                                    'FusedLayerType:FusedNode_conv2d_2_bn_relu_2',
                                    'Add:add_layer'])
        self.unit_test.assertEqual(_metadata['scheduling_info']['max_cut'], 256 * 3)

        expected_fused_nodes_mapping = {
            'conv2d_1_bn': 'FusedNode_conv2d_1_bn_relu_1',
            'relu_1': 'FusedNode_conv2d_1_bn_relu_1',
            'conv2d_2_bn': 'FusedNode_conv2d_2_bn_relu_2',
            'relu_2': 'FusedNode_conv2d_2_bn_relu_2'
        }
        self.unit_test.assertEqual(_metadata['scheduling_info']['fused_nodes_mapping'], expected_fused_nodes_mapping)



