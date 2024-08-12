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
import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class Activation16BitTest(BaseKerasFeatureNetworkTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, 'v4')
        # Force Mul base_config to 16bit only
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        mul_op_set.qc_options.base_config = [l for l in mul_op_set.qc_options.quantization_config_list if l.activation_n_bits == 16][0]
        tpc.layer2qco[tf.multiply].base_config = mul_op_set.qc_options.base_config
        return tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.multiply(inputs, inputs)
        x = tf.concat([x, x], axis=1)
        x = tf.add(x, np.ones((3,), dtype=np.float32))
        x1 = tf.subtract(x, np.ones((3,), dtype=np.float32))
        x = tf.multiply(x, x1)
        x = tf.keras.layers.Conv2D(3, 1)(x)
        outputs = tf.divide(x, 2*np.ones((3,), dtype=np.float32))
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        mul1_act_quant = quantized_model.layers[3]
        mul2_act_quant = quantized_model.layers[11]
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 16,
                                  "1st mul activation bits should be 16 bits because of following concat node.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == True,
                                  "1st mul activation should be forced by TPC to be signed, even though activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")


class Activation16BitMixedPrecisionTest(Activation16BitTest):

    def get_tpc(self):
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, 'v4')
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        mul_op_set.qc_options.base_config = [l for l in mul_op_set.qc_options.quantization_config_list if l.activation_n_bits == 16][0]
        tpc.layer2qco[tf.multiply].base_config = mul_op_set.qc_options.base_config
        mul_op_set.qc_options.quantization_config_list.extend(
            [mul_op_set.qc_options.base_config.clone_and_edit(activation_n_bits=4),
             mul_op_set.qc_options.base_config.clone_and_edit(activation_n_bits=2)])
        tpc.layer2qco[tf.multiply].quantization_config_list.extend([
            tpc.layer2qco[tf.multiply].base_config.clone_and_edit(activation_n_bits=4),
            tpc.layer2qco[tf.multiply].base_config.clone_and_edit(activation_n_bits=2)])

        return tpc

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(activation_memory=200)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.multiply(inputs, inputs)
        x = tf.add(x, np.ones((3,), dtype=np.float32))
        x1 = tf.subtract(x, np.ones((3,), dtype=np.float32))
        x = tf.multiply(x, x1)
        x = tf.keras.layers.Conv2D(3, 1)(x)
        outputs = tf.divide(x, 2*np.ones((3,), dtype=np.float32))
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        mul1_act_quant = quantized_model.layers[3]
        mul2_act_quant = quantized_model.layers[9]
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "1st mul activation bits should be 8 bits because of RU.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == False,
                                  "1st mul activation should be unsigned, because activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")
