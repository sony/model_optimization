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
import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.network_editors.actions import ChangeCandidatesActivationQuantConfigAttr, \
    ChangeQuantizationParamFunction, EditRule, ChangeCandidatesWeightsQuantConfigAttr
from model_compression_toolkit.core.common.network_editors.node_filters import NodeNameFilter, NodeNameScopeFilter, \
    NodeTypeFilter
from model_compression_toolkit.core.common.quantization.quantization_params_fn_selection import \
    get_weights_quantization_params_fn, get_activation_quantization_params_fn
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.v4.tpc_keras import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def get_uniform_weights(kernel, in_channels, out_channels):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [kernel, kernel, in_channels, out_channels])


class ScopeFilterTest(BaseKerasFeatureNetworkTest):
    '''
    - Check filter order- that the last filter overrides the one before it
    - Check scope filter
    - Check attribute changes
    '''

    def __init__(self, unit_test, activation_n_bits: int = 3, weights_n_bits: int = 3):
        self.activation_n_bits = activation_n_bits
        self.weights_n_bits = weights_n_bits
        self.kernel = 3
        self.num_conv_channels = 4
        self.scope = 'scope'
        self.conv_w = get_uniform_weights(self.kernel, self.num_conv_channels, self.num_conv_channels)
        super().__init__(unit_test, experimental_exporter=True)

    def get_tpc(self):
        tp_model = generate_test_tp_model({
            'weights_quantization_method': tp.QuantizationMethod.POWER_OF_TWO,
            'activation_n_bits': 16,
            'weights_n_bits': 16})
        return generate_keras_tpc(name="scope_filter_test", tp_model=tp_model)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.MSE,
                                      False, False, True)

    def get_debug_config(self):
        # first rule is to check that the scope filter catches the 2 convs with
        # second and third rule- they both do opperations on the same node.The goels are:
        #   1- to check "or" opperation. 2- to see that the last rule in the list is the last rule applied
        network_editor = [EditRule(filter=NodeNameScopeFilter(self.scope),
                                   action=ChangeCandidatesActivationQuantConfigAttr(activation_n_bits=self.activation_n_bits)),
                          EditRule(filter=NodeNameScopeFilter(self.scope),
                                   action=ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=self.weights_n_bits)),
                          EditRule(filter=NodeNameScopeFilter('2'),
                                   action=ChangeCandidatesWeightsQuantConfigAttr(enable_weights_quantization=True)),
                          EditRule(filter=NodeNameScopeFilter('2') or NodeNameScopeFilter('does_not_exist'),
                                   action=ChangeCandidatesWeightsQuantConfigAttr(enable_weights_quantization=False))
                          ]
        return mct.core.DebugConfig(network_editor=network_editor)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, self.num_conv_channels]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name='unchanged')(inputs)
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.scope + '_1')(x)
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.scope + '_2')(x)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # set conv weights to be integers uniformly distributed between
        # -(kernel*kernel*num_conv_channels*num_conv_channels)/2 : +(
        # kernel*kernel*num_conv_channels*num_conv_channels)/2
        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        model.layers[3].set_weights([self.conv_w])
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        # check that this conv's weights had changed due to change in number of bits
        self.unit_test.assertTrue(
            len(np.unique(conv_layers[2].get_quantized_weights()['kernel'].numpy())) in [2 ** (self.weights_n_bits) - 1,
                                                                             2 ** (self.weights_n_bits)])
        # check that this conv's weights did not change
        self.unit_test.assertTrue(np.all(conv_layers[0].get_quantized_weights()['kernel'].numpy() == self.conv_w))
        # check that this conv's weights did not change
        self.unit_test.assertTrue(np.all(conv_layers[2].layer.kernel == self.conv_w))
        self.unit_test.assertTrue(conv_layers[0].activation_quantizers[0].get_config()['num_bits'] == 16)
        self.unit_test.assertTrue(
            conv_layers[1].activation_quantizers[0].get_config()['num_bits'] == self.activation_n_bits)
        self.unit_test.assertTrue(
            conv_layers[2].activation_quantizers[0].get_config()['num_bits'] == self.activation_n_bits)


class NameFilterTest(BaseKerasFeatureNetworkTest):
    '''
    - Check name filter- that only the node with the name changed
    - Check the attribute change action on num weight bits and activation bits
    '''

    def __init__(self, unit_test, activation_n_bits: int = 3, weights_n_bits: int = 3):
        self.node_to_change_name = 'conv_to_change'
        self.activation_n_bits = activation_n_bits
        self.weights_n_bits = weights_n_bits
        self.kernel = 3
        self.num_conv_channels = 4
        # set conv weights to be integers uniformly distributed between
        # -(kernel*kernel*num_conv_channels*num_conv_channels)/2 : +(
        # kernel*kernel*num_conv_channels*num_conv_channels)/2
        self.conv_w = get_uniform_weights(self.kernel, self.num_conv_channels, self.num_conv_channels)
        super().__init__(unit_test, experimental_exporter=True)

    def get_tpc(self):
        tp_model = generate_test_tp_model({
            'weights_quantization_method': tp.QuantizationMethod.POWER_OF_TWO,
            'activation_n_bits': 16,
            'weights_n_bits': 16})
        return generate_keras_tpc(name="name_filter_test", tp_model=tp_model)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.MSE,
                                      False, False, True)

    def get_debug_config(self):
        network_editor = [EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                   action=ChangeCandidatesActivationQuantConfigAttr(activation_n_bits=self.activation_n_bits)),
                          EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                   action=ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=self.weights_n_bits))
                          ]
        return mct.core.DebugConfig(network_editor=network_editor)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, self.num_conv_channels]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.node_to_change_name)(inputs)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        # check that this conv's weights had changed due to change in number of bits
        self.unit_test.assertTrue(
            len(np.unique(conv_layers[0].get_quantized_weights()['kernel'].numpy())) in [2 ** (self.weights_n_bits) - 1,
                                                                             2 ** (self.weights_n_bits)])
        # check that this conv's weights did not change
        self.unit_test.assertTrue(np.all(conv_layers[1].get_quantized_weights()['kernel'].numpy() == self.conv_w))
        self.unit_test.assertTrue(conv_layers[0].activation_quantizers[0].get_config()['num_bits'] == self.activation_n_bits)
        self.unit_test.assertTrue(conv_layers[1].activation_quantizers[0].get_config()['num_bits'] == 16)


class TypeFilterTest(BaseKerasFeatureNetworkTest):
    '''
    - Check node type filter
    - Check threshold function action
    - Check "and" between filters
    '''

    def __init__(self, unit_test, activation_n_bits: int = 3, weights_n_bits: int = 3):
        self.node_to_change_name = 'conv_to_change'
        self.type_to_change = layers.Conv2D
        self.activation_n_bits = activation_n_bits
        self.weights_n_bits = weights_n_bits
        self.kernel = 3
        self.num_conv_channels = 4
        self.conv_w = np.random.uniform(0, 1,
                                        [self.kernel, self.kernel, self.num_conv_channels, self.num_conv_channels])
        # set a weight above 1
        self.conv_w[0, 0, 0, 0] = 1.1
        super().__init__(unit_test, experimental_exporter=True)

    def weights_params_fn(self):
        return get_weights_quantization_params_fn(tp.QuantizationMethod.POWER_OF_TWO)

    def activations_params_fn(self):
        return get_activation_quantization_params_fn(tp.QuantizationMethod.POWER_OF_TWO)

    def get_tpc(self):
        tp_model = generate_test_tp_model({
            'weights_quantization_method': tp.QuantizationMethod.POWER_OF_TWO,
            'activation_n_bits': 16,
            'weights_n_bits': 16})
        return generate_keras_tpc(name="type_filter_test", tp_model=tp_model)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.MSE,
                                      False, False, False)

    def get_debug_config(self):
        network_editor = [EditRule(filter=NodeTypeFilter(self.type_to_change),
                                   action=ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=self.weights_n_bits)),
                          EditRule(filter=NodeTypeFilter(self.type_to_change),
                                   action=ChangeCandidatesActivationQuantConfigAttr(activation_n_bits=self.activation_n_bits)),
                          EditRule(filter=NodeTypeFilter(self.type_to_change).__and__(NodeNameFilter(self.node_to_change_name)),
                                   action=ChangeQuantizationParamFunction(weights_quantization_params_fn=self.weights_params_fn())),
                          EditRule(filter=NodeTypeFilter(self.type_to_change).__and__(NodeNameFilter(self.node_to_change_name)),
                                   action=ChangeQuantizationParamFunction(
                                       activation_quantization_params_fn=self.activations_params_fn())),
                          EditRule(filter=NodeNameFilter(self.node_to_change_name) and NodeTypeFilter(layers.ReLU),
                                   action=ChangeCandidatesActivationQuantConfigAttr(activation_n_bits=16))]
        return mct.core.DebugConfig(network_editor=network_editor)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 224, self.num_conv_channels]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.node_to_change_name)(inputs)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # set conv weights to be integers uniformly distributed between
        # -(kernel*kernel*num_conv_channels*num_conv_channels)/2 : +(
        # kernel*kernel*num_conv_channels*num_conv_channels)/2
        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        # check that the two conv in the network same weights.
        self.unit_test.assertTrue(conv_layers[0].get_quantized_weights()['kernel'].numpy().max() == quantized_model.layers[3].get_quantized_weights()['kernel'].numpy().max())
        self.unit_test.assertTrue(conv_layers[0].activation_quantizers[0].get_config()['num_bits'] == self.activation_n_bits)
        self.unit_test.assertTrue(conv_layers[1].activation_quantizers[0].get_config()['num_bits'] == self.activation_n_bits)
