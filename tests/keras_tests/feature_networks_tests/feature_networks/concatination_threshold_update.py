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


import tensorflow as tf

from model_compression_toolkit.core.keras.constants import THRESHOLD
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from model_compression_toolkit.core import QuantizationConfig
import numpy as np

"""
this checks that thresold prior to concat have been updated correctly.
"""

keras = tf.keras
layers = keras.layers


class ConcatThresholdtest(BaseKerasFeatureNetworkTest):
    """
    This tests that all thresholds are equal
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return QuantizationConfig(concat_threshold_update=True)
    
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        pre_concat_1 = layers.Conv2D(3, 4)(inputs)
        pre_concat_2 = layers.Conv2D(3, 4)(inputs)
        outputs = layers.Concatenate(axis=-1)([pre_concat_1, 8*pre_concat_2])
        return keras.Model(inputs=inputs, outputs=outputs)
    

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):

        conv1_threshold = quantized_model.layers[6].activation_holder_quantizer.get_config()['threshold'][0]
        conv2_threshold = quantized_model.layers[7].activation_holder_quantizer.get_config()['threshold'][0]
        concat_threshold = quantized_model.layers[9].activation_holder_quantizer.get_config()['threshold'][0]

        self.unit_test.assertTrue(conv1_threshold == concat_threshold and conv2_threshold == concat_threshold)
