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

import tensorflow as tf
import numpy as np

import model_compression_toolkit as mct
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from mct_quantizers.keras.metadata import add_metadata, get_metadata

from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class MetadataTest(BaseKerasFeatureNetworkTest):

    def get_tpc(self):
        return mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v2")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        output = tf.add(inputs, inputs)
        return tf.keras.models.Model(inputs=inputs, outputs=output)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(get_metadata(quantized_model)) > 0,
                                  msg='A model quantized with TPC IMX500.v2 should have a metadata.')
