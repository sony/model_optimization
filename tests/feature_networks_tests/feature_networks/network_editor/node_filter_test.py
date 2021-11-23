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


from model_compression_toolkit.common.matchers.node_matcher import NodeAndMatcher
from model_compression_toolkit.common.quantization.quantization_params_fn_selection import \
    get_weights_quantization_params_fn
from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity
from model_compression_toolkit.common.network_editors.node_filters import NodeNameFilter, NodeNameScopeFilter, \
    NodeTypeFilter
from model_compression_toolkit.common.network_editors.actions import ChangeActivationQuantConfigAttr, \
    ChangeQuantizationParamFunction, EditRule, ChangeCandidatesWeightsQuantConfigAttr, ChangeFinalWeightsQuantConfigAttr

keras = tf.keras
layers = keras.layers


def get_uniform_weights(kernel, in_channels, out_channels):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [kernel, kernel, in_channels, out_channels])


class ScopeFilterTest(BaseFeatureNetworkTest):
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
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE, mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO, 16, 16,
                                      False, False, True)

    def get_network_editor(self):
        # first rule is to check that the scope filter catches the 2 convs with
        # second and third rule- they both do opperations on the same node.The goels are:
        #   1- to check "or" opperation. 2- to see that the last rule in the list is the last rule applied
        return [EditRule(filter=NodeNameScopeFilter(self.scope),
                         action=ChangeActivationQuantConfigAttr(activation_n_bits=self.activation_n_bits)),
                EditRule(filter=NodeNameScopeFilter(self.scope),
                         action=ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=self.weights_n_bits)),
                EditRule(filter=NodeNameScopeFilter('2'),
                         action=ChangeCandidatesWeightsQuantConfigAttr(enable_weights_quantization=True)),
                EditRule(filter=NodeNameScopeFilter('2') or NodeNameScopeFilter('does_not_exist'),
                         action=ChangeCandidatesWeightsQuantConfigAttr(enable_weights_quantization=False))
                ]

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, self.num_conv_channels]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
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
        # check that this conv's weights had changed due to change in number of bits
        self.unit_test.assertTrue(
            len(np.unique(quantized_model.layers[4].weights[0].numpy())) in [2 ** (self.weights_n_bits) - 1,
                                                                             2 ** (self.weights_n_bits)])
        # check that this conv's weights did not change
        self.unit_test.assertTrue(np.all(quantized_model.layers[2].weights[0].numpy() == self.conv_w))
        # check that this conv's weights did not change
        self.unit_test.assertTrue(np.all(quantized_model.layers[6].weights[0].numpy() == self.conv_w))
        self.unit_test.assertTrue(quantized_model.layers[3].inbound_nodes[0].call_kwargs['num_bits'] == 16)
        self.unit_test.assertTrue(
            quantized_model.layers[5].inbound_nodes[0].call_kwargs['num_bits'] == self.activation_n_bits)
        self.unit_test.assertTrue(
            quantized_model.layers[7].inbound_nodes[0].call_kwargs['num_bits'] == self.activation_n_bits)


class NameFilterTest(BaseFeatureNetworkTest):
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
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE, mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO, 16, 16,
                                      False, False, True)

    def get_network_editor(self):
        return [EditRule(filter=NodeNameFilter(self.node_to_change_name),
                         action=ChangeActivationQuantConfigAttr(activation_n_bits=self.activation_n_bits)),
                EditRule(filter=NodeNameFilter(self.node_to_change_name),
                         action=ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=self.weights_n_bits))
                ]

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, self.num_conv_channels]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.node_to_change_name)(inputs)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that this conv's weights had changed due to change in number of bits
        self.unit_test.assertTrue(
            len(np.unique(quantized_model.layers[2].weights[0].numpy())) in [2 ** (self.weights_n_bits) - 1,
                                                                             2 ** (self.weights_n_bits)])
        # check that this conv's weights did not change
        self.unit_test.assertTrue(np.all(quantized_model.layers[4].weights[0].numpy() == self.conv_w))
        self.unit_test.assertTrue(
            quantized_model.layers[3].inbound_nodes[0].call_kwargs['num_bits'] == self.activation_n_bits)
        self.unit_test.assertTrue(quantized_model.layers[5].inbound_nodes[0].call_kwargs['num_bits'] == 16)


class TypeFilterTest(BaseFeatureNetworkTest):
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
        super().__init__(unit_test)

    def params_fn(self):
        return get_weights_quantization_params_fn(mct.QuantizationMethod.POWER_OF_TWO,
                                                  mct.ThresholdSelectionMethod.NOCLIPPING)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE, mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, False, False, False)

    def get_network_editor(self):
        return [EditRule(filter=NodeTypeFilter(self.type_to_change),
                         action=ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=self.weights_n_bits)),
                EditRule(filter=NodeTypeFilter(self.type_to_change),
                         action=ChangeActivationQuantConfigAttr(activation_n_bits=self.activation_n_bits)),
                EditRule(filter=NodeTypeFilter(self.type_to_change).__and__(NodeNameFilter(self.node_to_change_name)),
                         action=ChangeQuantizationParamFunction(weights_quantization_params_fn=self.params_fn())),
                EditRule(filter=NodeNameFilter(self.node_to_change_name) and NodeTypeFilter(layers.ReLU),
                         action=ChangeActivationQuantConfigAttr(activation_n_bits=16))]

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 224, self.num_conv_channels]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
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
        # check that the two conv in the network have different weights. In order for this to happen, their weight's num
        # bits needed to change, and one of the conv's threshold function needed to change to 'no_clipping'
        self.unit_test.assertTrue(
            quantized_model.layers[2].weights[0].numpy().max() != quantized_model.layers[4].weights[0].numpy().max())
        self.unit_test.assertTrue(
            quantized_model.layers[3].inbound_nodes[0].call_kwargs['num_bits'] == self.activation_n_bits)
        self.unit_test.assertTrue(
            quantized_model.layers[5].inbound_nodes[0].call_kwargs['num_bits'] == self.activation_n_bits)


class FilterLogicTest(BaseFeatureNetworkTest):
    '''
    - Check "and" and "or" operations between filters
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
        super().__init__(unit_test)

    def params_fn(self):
        return get_weights_quantization_params_fn(cmo.QuantizationMethod.POWER_OF_TWO,
                                                  cmo.ThresholdSelectionMethod.NOCLIPPING)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE, mct.ThresholdSelectionMethod.MSE,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16,
                                      False, False, False)

    def get_network_editor(self):
        return [(NodeTypeFilter(self.type_to_change),
                 ChangeQuantConfigAttr(weights_n_bits=self.weights_n_bits, activation_n_bits=self.activation_n_bits)),
                (NodeAndMatcher(NodeTypeFilter(self.type_to_change), NodeNameFilter(self.node_to_change_name)),
                 ChangeQuantizationParamFunction(weights_quantization_params_fn=self.params_fn())),
                (NodeAndMatcher(NodeTypeFilter(layers.ReLU), NodeNameFilter(self.node_to_change_name)),
                 ChangeActivationQuantConfigAttr(activation_n_bits=16))
                ]

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 224, self.num_conv_channels]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
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
        # check that the two conv in the network have different weights. In order for this to happen, their weight's num
        # bits needed to change, and one of the conv's threshold function needed to change to 'no_clipping'
        self.unit_test.assertTrue(
            quantized_model.layers[2].weights[0].numpy().max() != quantized_model.layers[4].weights[0].numpy().max())
