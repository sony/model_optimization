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

import model_compression_toolkit as mct
from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.core.common.network_editors.actions import EditRule, \
    ChangeCandidatesWeightsQuantizationMethod
from model_compression_toolkit.core.common.network_editors.node_filters import NodeNameFilter
from model_compression_toolkit.core.keras.constants import KERNEL
from mct_quantizers.keras.quantizers import ActivationLutPOTInferableQuantizer
from mct_quantizers.common.constants import THRESHOLD, LUT_VALUES
from tests.common_tests.helpers.generate_test_tp_model import generate_test_attr_configs, generate_test_op_qc, \
    generate_test_tp_model
from tests.keras_tests.exporter_tests.tflite_int8.imx500_int8_tp_model import generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


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
        super().__init__(unit_test, num_calibration_iter=5, val_batch_size=32, experimental_exporter=True)

    def get_tpc(self):
        qmethod = tp.QuantizationMethod.LUT_SYM_QUANTIZER if self.is_symmetric else tp.QuantizationMethod.LUT_POT_QUANTIZER
        tp_model = generate_test_tp_model({'weights_n_bits': self.weights_n_bits,
                                           'weights_quantization_method': qmethod})
        return generate_keras_tpc(name='lut_quantizer_test', tp_model=tp_model)

    def get_debug_config(self):
        return mct.core.DebugConfig(
            network_editor=[EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                     action=ChangeCandidatesWeightsQuantizationMethod(
                                         weights_quantization_method=
                                         mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                                         attr_name=KERNEL))])

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
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        self.unit_test.assertTrue(np.sum(
            np.abs(conv_layers[0].get_quantized_weights()[KERNEL]) - conv_layers[1].get_quantized_weights()[KERNEL]) > 0)


class LUTActivationQuantizerTest(BaseKerasFeatureNetworkTest):
    '''
    - Check that activation are quantized correctly using LUT quantizer
    '''

    def __init__(self, unit_test, activation_n_bits: int = 3):
        self.activation_n_bits = activation_n_bits
        self.num_conv_channels = 4
        self.kernel = 3
        super().__init__(unit_test, num_calibration_iter=5, val_batch_size=32, experimental_exporter=True)

    def get_tpc(self):
        tp_model = generate_test_tp_model({'activation_quantization_method': tp.QuantizationMethod.LUT_POT_QUANTIZER,
                                           'activation_n_bits': self.activation_n_bits})
        return generate_keras_tpc(name='lut_quantizer_test', tp_model=tp_model)

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
        all_expected_lut_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)

        for ll in all_expected_lut_layers:
            # Check that lut quantizer layer is added where expected (after each layer, for quantizing activation)
            self.unit_test.assertTrue(isinstance(ll.activation_holder_quantizer, ActivationLutPOTInferableQuantizer))
            # Check layer's thresholds are power of two
            self.unit_test.assertTrue(math.log2(ll.activation_holder_quantizer.get_config()[THRESHOLD][0]).is_integer())
            # Check layers number of lut values
            self.unit_test.assertTrue(len(ll.activation_holder_quantizer.get_config()[LUT_VALUES]) <= 2 ** self.activation_n_bits)
            self.unit_test.assertTrue(np.all(np.mod(ll.activation_holder_quantizer.get_config()[LUT_VALUES], 1) == 0))


if __name__ == '__main__':
    unittest.main()
