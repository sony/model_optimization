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
import math
import unittest

import numpy as np
import tensorflow as tf

import model_compression_toolkit as cmo
from model_compression_toolkit.core.common.constants import EPS, \
    MULTIPLIER_N_BITS
from model_compression_toolkit.core.common.network_editors.actions import EditRule, \
    ChangeCandidatesWeightsQuantizationMethod
from model_compression_toolkit.core.common.network_editors.node_filters import NodeNameFilter
from model_compression_toolkit.core.keras.quantizer.lut_fake_quant import LUTFakeQuant
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers
tp = cmo.target_platform


def lut_fake_quant_op(input_data,
                      activation_is_signed,
                      cluster_centers,
                      threshold):
    if activation_is_signed is None or cluster_centers is None or threshold is None:
        return None

    _quant_output = lut_kmeans_quantizer(input_data,
                                         activation_is_signed,
                                         cluster_centers,
                                         threshold)
    return _quant_output


def lut_kmeans_quantizer(tensor_data,
                         activation_is_signed,
                         cluster_centers,
                         threshold):
    tensor = int_quantization_with_threshold(tensor_data, MULTIPLIER_N_BITS,
                                             activation_is_signed,
                                             threshold)
    tensor = tf.expand_dims(tensor, -1)

    expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
    cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
    centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

    quant_tensor = (centers / (2 ** (MULTIPLIER_N_BITS - int(activation_is_signed)))) * threshold

    return quant_tensor


def int_quantization_with_threshold(data,
                                    n_bits,
                                    activation_is_signed,
                                    threshold,
                                    eps=EPS):
    if activation_is_signed:
        clip_max = 2 ** (n_bits - 1) - 1
        clip_min = -2 ** (n_bits - 1)
    else:
        clip_max = 2 ** n_bits - 1
        clip_min = 0

    return tf.clip_by_value((data / (threshold + eps)) * (2 ** (n_bits - int(activation_is_signed))),
                            clip_value_max=clip_max, clip_value_min=clip_min)


def get_uniform_weights(kernel, in_channels, out_channels):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [out_channels, kernel, kernel, in_channels]).transpose(1, 2, 3, 0)


class LUTWeightsQuantizerTest(BaseKerasFeatureNetworkTest):
    '''
    - Check name filter- that only the node with the name changed
    - Check that different quantization methods on the same weights give different results
    '''

    def __init__(self, unit_test, weights_n_bits: int = 3, is_symmetric=False):
        self.weights_n_bits = weights_n_bits
        self.node_to_change_name = 'change'
        self.num_conv_channels = 4
        self.kernel = 3
        self.conv_w = get_uniform_weights(self.kernel, self.num_conv_channels, self.num_conv_channels)
        self.is_symmetric = is_symmetric
        super().__init__(unit_test, num_calibration_iter=5, val_batch_size=32)

    def get_tpc(self):
        qmethod = tp.QuantizationMethod.LUT_SYM_QUANTIZER if self.is_symmetric else tp.QuantizationMethod.LUT_POT_QUANTIZER
        qco = tp.QuantizationConfigOptions(
            [tp.OpQuantizationConfig(activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
                                     weights_quantization_method=qmethod,
                                     activation_n_bits=8,
                                     weights_n_bits=self.weights_n_bits,
                                     weights_per_channel_threshold=True,
                                     enable_weights_quantization=True,
                                     enable_activation_quantization=True,
                                     quantization_preserving=False,
                                     fixed_scale=None,
                                     fixed_zero_point=None,
                                     weights_multiplier_nbits=None
                                     )])
        return tp.TargetPlatformCapabilities(tp.TargetPlatformModel(qco))

    def get_quantization_config(self):
        return cmo.QuantizationConfig()

    def get_network_editor(self):
        return [EditRule(filter=NodeNameFilter(self.node_to_change_name),
                         action=ChangeCandidatesWeightsQuantizationMethod(
                             weights_quantization_method=cmo.target_platform.QuantizationMethod.POWER_OF_TWO))]

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, self.num_conv_channels]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.node_to_change_name)(inputs)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        self.unit_test.assertTrue(np.sum(
            np.abs(quantized_model.layers[2].weights[0].numpy()) - quantized_model.layers[4].weights[0].numpy()) > 0)


class LUTActivationQuantizerTest(BaseKerasFeatureNetworkTest):
    '''
    - Check that activation are quantized correctly using LUT quantizer
    '''

    def __init__(self, unit_test, activation_n_bits: int = 3):
        self.activation_n_bits = activation_n_bits
        self.num_conv_channels = 4
        self.kernel = 3
        super().__init__(unit_test, num_calibration_iter=5, val_batch_size=32)

    def get_tpc(self):
        qco = tp.QuantizationConfigOptions(
            [tp.OpQuantizationConfig(activation_quantization_method=tp.QuantizationMethod.LUT_POT_QUANTIZER,
                                     weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
                                     activation_n_bits=self.activation_n_bits,
                                     weights_n_bits=8,
                                     weights_per_channel_threshold=True,
                                     enable_weights_quantization=True,
                                     enable_activation_quantization=True,
                                     quantization_preserving=False,
                                     fixed_scale=None,
                                     fixed_zero_point=None,
                                     weights_multiplier_nbits=None
                                     )])
        return tp.TargetPlatformCapabilities(tp.TargetPlatformModel(qco))

    def get_quantization_config(self):
        return cmo.QuantizationConfig()

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, self.num_conv_channels]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(inputs)
        x = layers.ReLU()(x)
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        all_expected_lut_layers = np.array(quantized_model.layers)[
            [i for i in range(1, len(quantized_model.layers), 2)]]

        for ll in all_expected_lut_layers:
            # Check that lut quantizer layer is added where expected (after each layer, for quantizing activation)
            self.unit_test.assertTrue(isinstance(ll, LUTFakeQuant))
            # Check layer's thresholds are power of two
            self.unit_test.assertTrue(math.log2(ll.threshold).is_integer())
            # Check layers number of clusters and clusters values
            self.unit_test.assertTrue(ll.cluster_centers.shape[0] <= 2 ** self.activation_n_bits)
            self.unit_test.assertTrue(np.all(np.mod(ll.cluster_centers, 1) == 0))

            # Check the output of each LUT Fake Quant is as expected
            ll_output = ll(input_x[0])
            expected_output = lut_fake_quant_op(input_x[0],
                                                ll.activation_is_signed,
                                                ll.cluster_centers,
                                                ll.threshold)
            self.unit_test.assertTrue(np.all(ll_output == expected_output))


class RunKmeansTest(unittest.TestCase):
    def test_lut_quantizer(self):
        LUTWeightsQuantizerTest(self).run_test()
        LUTWeightsQuantizerTest(self, is_symmetric=True).run_test()


if __name__ == '__main__':
    unittest.main()
