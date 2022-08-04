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


import tensorflow as tf
import numpy as np

from model_compression_toolkit.core.common.network_editors import EditRule, NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.network_editors.actions import ReplaceLayer
from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers


# define custom layer as a relu replacement
class Identity(keras.layers.Layer):
    def __init__(self, **kwargs):
        super_kwargs = {'name': kwargs.get('name'),
                        'dtype': kwargs.get('dtype')
                        }
        super(Identity, self).__init__(**super_kwargs)

    def call(self, inputs):
        return inputs


# modify the config and weights for the new layer (no change is required)
def get_identity_params_from_relu(weights={}, **kwargs):
    return weights, kwargs


# Test1: replacing a single Relu layer with identity layer
class SingleReluReplacementTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return get_quantization_disabled_keras_tpc("single_relu_replacement_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.ReLU(name='ReLU_1')(inputs)
        outputs = layers.ReLU(max_value=6, name='ReLU_2')(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], Identity))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.ReLU))

    def get_network_editor(self):
        #   replace all Relu_1 with identity custom layer
        return [EditRule(filter=NodeNameFilter('ReLU_1'),
                         action=ReplaceLayer(Identity, get_identity_params_from_relu))
                ]


# Test2: replacing all Relu layers with identity layers
class ReluReplacementTest(SingleReluReplacementTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(np.isclose(0, np.mean(quantized_model.predict(input_x) - input_x)))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], Identity))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], Identity))

    def get_network_editor(self):
        #   replace all Relu's with identity custom layer
        return [EditRule(filter=NodeTypeFilter(layers.ReLU),
                         action=ReplaceLayer(Identity, get_identity_params_from_relu))
                ]


# define custom layer as a relu replacement
class AddBias(keras.layers.Layer):
    def __init__(self, bias, **kwargs):
        super_kwargs = {'name': kwargs.get('name'),
                        'dtype': kwargs.get('dtype')
                        }
        super(AddBias, self).__init__(**super_kwargs)
        self.bias = bias

    def call(self, inputs):
        return inputs + self.bias


# modify the config and weights for the new layer (no change is required)
def get_add_bias_params_from_relu(weights={}, **kwargs):
    if kwargs.get('max_value') is None:
        bias = 0
    else:
        bias = kwargs.get('max_value')

    kwargs.update({'bias': bias})
    return weights, kwargs


# Test3: replacing all Relu layers with AddBias layers
class ReluReplacementWithAddBiasTest(SingleReluReplacementTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(np.isclose(6, np.mean(quantized_model.predict(input_x) - input_x)))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], AddBias))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], AddBias))
        self.unit_test.assertTrue(quantized_model.layers[1].bias == 0)
        self.unit_test.assertTrue(quantized_model.layers[2].bias == 6)

    def get_network_editor(self):
        #   replace all Relu's with identity custom layer
        return [EditRule(filter=NodeTypeFilter(layers.ReLU),
                         action=ReplaceLayer(AddBias, get_add_bias_params_from_relu))
                ]
