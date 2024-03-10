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

import numpy as np
import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.input_layer import InputLayer
else:
    from keras.engine.input_layer import InputLayer

from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.core import QuantizationErrorMethod, DebugConfig
from model_compression_toolkit.core.common.network_editors.actions import EditRule, \
    ChangeCandidatesActivationQuantConfigAttr
from model_compression_toolkit.core.common.network_editors.node_filters import NodeTypeFilter
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class EditActivationErrorMethod(BaseKerasFeatureNetworkTest):
    """
    Test the change of activation error method.
    Use inputs with many values around 0.2 except for a single value of 2.
    In MSE the threshold is 1, but in NOCLIPPING is 2.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test,
                         input_shape=(224, 224, 3))

    def generate_inputs(self):
        input_data = [np.full(shape=in_shape, fill_value=0.2) for in_shape in self.get_input_shapes()]
        input_data[0][0, 0, 0, 0] = 2
        return input_data

    def get_debug_config(self):
        return DebugConfig(network_editor=[EditRule(filter=NodeTypeFilter(InputLayer),
                                                    action=ChangeCandidatesActivationQuantConfigAttr(
                                                        activation_error_method=QuantizationErrorMethod.NOCLIPPING))])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4, use_bias=False)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[0]
        input_q_params = holder_layer.activation_holder_quantizer.get_config()
        threshold = input_q_params['threshold']
        self.unit_test.assertTrue(len(threshold) == 1, f'In per-tensor quantization, expected to find a single threshold but found {len(threshold)} thresholds')
        self.unit_test.assertTrue(threshold[0] == 2,
                                  f'After editing input layer to no clipping error method,'
                                  f'threshold should be 2, but is {threshold}')
