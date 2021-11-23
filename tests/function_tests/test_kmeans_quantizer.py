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

import unittest

from model_compression_toolkit.common.network_editors.node_filters import NodeNameFilter
from model_compression_toolkit.common.network_editors.actions import EditRule, ChangeCandidatesWeightsQuantConfigAttr, \
    ChangeActivationQuantConfigAttr, \
    ChangeFinalWeightsQuantConfigAttr
from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as cmo
import tensorflow as tf
import numpy as np

keras = tf.keras
layers = keras.layers


def get_uniform_weights(kernel, in_channels, out_channels):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [out_channels, kernel, kernel, in_channels]).transpose(1, 2, 3, 0)


def get_zero_as_weights(kernel, in_channels, out_channels):
    return np.zeros([kernel, kernel, in_channels, out_channels])


class KmeansQuantizerTestBase(BaseFeatureNetworkTest):
    '''
    - Check name filter- that only the node with the name changed
    - Check that different quantization methods on the same weights give different results
    '''

    def __init__(self, unit_test, quantization_method: cmo.QuantizationMethod.KMEANS, weight_fn=get_uniform_weights,
                 weights_n_bits: int = 3):
        self.quantization_method = quantization_method
        self.weights_n_bits = weights_n_bits
        self.node_to_change_name = 'change'
        self.num_conv_channels = 4
        self.kernel = 3
        self.conv_w = weight_fn(self.kernel, self.num_conv_channels, self.num_conv_channels)
        super().__init__(unit_test, num_calibration_iter=5, val_batch_size=32)

    def get_quantization_config(self):
        return cmo.QuantizationConfig(cmo.ThresholdSelectionMethod.MSE, cmo.ThresholdSelectionMethod.MSE,
                                      cmo.QuantizationMethod.POWER_OF_TWO, self.quantization_method, 4,
                                      self.weights_n_bits,
                                      False, False, True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 16, 16, self.num_conv_channels]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.node_to_change_name)(inputs)
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        model.layers[3].set_weights([self.conv_w])
        return model

    def get_network_editor(self):
        return [EditRule(filter=NodeNameFilter(self.node_to_change_name),
                         action=ChangeCandidatesWeightsQuantConfigAttr(
                             weights_quantization_method=cmo.QuantizationMethod.POWER_OF_TWO))]

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        self.unit_test.assertTrue(np.sum(
            np.abs(quantized_model.layers[2].weights[0].numpy() - quantized_model.layers[4].weights[0].numpy())) > 0)


class KmeansQuantizerTest(KmeansQuantizerTestBase):
    '''
    This test checks the chosen quantization method is different that symmetric uniform
    '''

    def __init__(self, unit_test, quantization_method: cmo.QuantizationMethod.KMEANS, weights_n_bits: int = 3):
        super().__init__(unit_test, quantization_method, get_uniform_weights, weights_n_bits)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        self.unit_test.assertTrue(np.sum(
            np.abs(quantized_model.layers[2].weights[0].numpy() - quantized_model.layers[4].weights[0].numpy())) > 0)


class KmeansQuantizerTestManyClasses(KmeansQuantizerTestBase):
    '''
    This test checks the chosen quantization method is different that symmetric uniform
    '''

    def __init__(self, unit_test, quantization_method: cmo.QuantizationMethod.KMEANS, weights_n_bits: int = 8):
        super().__init__(unit_test, quantization_method, get_uniform_weights, weights_n_bits)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        self.unit_test.assertTrue(
            np.all(np.isclose(float_model.layers[1].weights[0].numpy(), quantized_model.layers[4].weights[0].numpy())))


class KmeansQuantizerTestZeroWeights(KmeansQuantizerTestBase):
    '''
    This test checks the case where all the weight values are zero
    '''

    def __init__(self, unit_test, quantization_method: cmo.QuantizationMethod.KMEANS, weights_n_bits: int = 3):
        super().__init__(unit_test, quantization_method, get_zero_as_weights, weights_n_bits)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        self.unit_test.assertTrue(np.sum(np.abs(quantized_model.layers[2].weights[0].numpy())) == 0)
        self.unit_test.assertTrue(np.sum(np.abs(quantized_model.layers[4].weights[0].numpy())) == 0)
        self.unit_test.assertTrue(np.sum(np.abs(quantized_model.layers[6].weights[0].numpy())) == 0)


# This test checks that the Kmeans quantization has a different result than symmetric uniform quantization
class RunKmeansTest(unittest.TestCase):
    def test_kmeans_quantizer(self):
        KmeansQuantizerTest(self, cmo.QuantizationMethod.KMEANS).run_test()


# This test checks that the LUT- Kmeans quantization has a different result than symmetric uniform quantization
class RunLutKmeansTest(unittest.TestCase):
    def test_kmeans_quantizer(self):
        KmeansQuantizerTest(self, cmo.QuantizationMethod.LUT_QUANTIZER).run_test()


# In this test we have weights with less unique values than the number of clusters
class RunKmeansTestManyClasses(unittest.TestCase):
    def test_kmeans_quantizer(self):
        KmeansQuantizerTestManyClasses(self, cmo.QuantizationMethod.KMEANS, weights_n_bits=8).run_test()


# In this test we have weights with less unique values than the number of clusters
class RunLutKmeansTestManyClasses(unittest.TestCase):
    def test_kmeans_quantizer(self):
        KmeansQuantizerTestManyClasses(self, cmo.QuantizationMethod.LUT_QUANTIZER, weights_n_bits=8).run_test()


# This test checks the case where all the weight values are zero
class RunKmeansTestZeroWeights(unittest.TestCase):
    def test_kmeans_quantizer_zero_weights(self):
        KmeansQuantizerTest(self, cmo.QuantizationMethod.KMEANS).run_test()


# This test checks the case where all the weight values are zero
class RunLutKmeansTestZeroWeights(unittest.TestCase):
    def test_kmeans_quantizer_zero_weights(self):
        KmeansQuantizerTest(self, cmo.QuantizationMethod.LUT_QUANTIZER).run_test()


if __name__ == '__main__':
    unittest.main()
