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
from keras import activations

from model_compression_toolkit.core.common.network_editors import EditRule, NodeTypeFilter
from model_compression_toolkit.core.common.network_editors.actions import ReplaceLayer
from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers


# define custom layer as a relu replacement
class Linear(keras.layers.Layer):
    def __init__(self, **kwargs):
        super_kwargs = {'name': kwargs.get('name'),
                        'dtype': kwargs.get('dtype')
                        }
        super(Linear, self).__init__(**super_kwargs)
        # self.bias = kwargs.bias

    def call(self, inputs):
        return inputs


# modify the config and weights for the new layer
def linear_config(weights=[], **kwargs):
    # config = kwargs
    # config.update({'bias': 1})
    return kwargs, weights


class SwishReplacementTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return get_quantization_disabled_keras_tpc("swish_replacement_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.nn.swish(inputs)
        outputs = tf.nn.swish(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(np.isclose(0,np.mean(quantized_model.predict(input_x) - input_x)))

    def get_network_editor(self):
        # first rule is to check that the scope filter catches the 2 convs with
        # second and third rule- they both do opperations on the same node.The goels are:
        #   1- to check "or" opperation. 2- to see that the last rule in the list is the last rule applied
        return [EditRule(filter=NodeTypeFilter(tf.nn.swish),
                         action=ReplaceLayer(Linear, linear_config))
                ]


