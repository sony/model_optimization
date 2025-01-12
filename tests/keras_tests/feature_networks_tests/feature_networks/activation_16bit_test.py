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
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TPC
from mct_quantizers.keras.activation_quantization_holder import KerasActivationQuantizationHolder
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set
from tests.common_tests.helpers.generate_test_tpc import generate_custom_test_tpc
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import get_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class Activation16BitTest(BaseKerasFeatureNetworkTest):

    def get_tpc(self):
        tpc = get_tpc()
        base_cfg_16 = [c for c in get_config_options_by_operators_set(tpc,
                                                                      OperatorSetNames.MUL).quantization_configurations
                       if c.activation_n_bits == 16][0].clone_and_edit()
        qco_16 = QuantizationConfigOptions(base_config=base_cfg_16,
                                           quantization_configurations=(tpc.default_qco.base_config,
                                                                        base_cfg_16))
        tpc = generate_custom_test_tpc(
            name="custom_16_bit_tpc",
            base_cfg=tpc.default_qco.base_config,
            base_tpc=tpc,
            operator_sets_dict={
                OperatorSetNames.MUL: qco_16,
            })

        return tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.multiply(inputs, inputs)
        x = tf.concat([x, x], axis=1)
        x = tf.add(x, np.ones((3,), dtype=np.float32))
        x1 = tf.subtract(x, np.ones((3,), dtype=np.float32))
        x = tf.multiply(x, x1)
        x = tf.reshape(x, (-1, 4, 4, 8, 3))
        x = tf.reshape(x, (-1, 16, 8, 3))
        x = tf.keras.layers.Conv2D(3, 1)(x)
        outputs = tf.divide(x, 2*np.ones((3,), dtype=np.float32))
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        act_quant_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        mul1_act_quant, mul2_act_quant = act_quant_layers[1], act_quant_layers[5]
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 16,
                                  "1st mul activation bits should be 16 bits because of following concat node.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == True,
                                  "1st mul activation should be forced by TPC to be signed, even though activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")


class Activation16BitMixedPrecisionTest(Activation16BitTest):

    def get_tpc(self):
        tpc = get_tpc()

        mul_qco = get_config_options_by_operators_set(tpc, OperatorSetNames.MUL)
        base_cfg_16 = [l for l in mul_qco.quantization_configurations if l.activation_n_bits == 16][0]
        quantization_configurations = list(mul_qco.quantization_configurations)
        quantization_configurations.extend([
            base_cfg_16.clone_and_edit(activation_n_bits=4),
            base_cfg_16.clone_and_edit(activation_n_bits=2)])

        qco_16 = QuantizationConfigOptions(base_config=base_cfg_16,
                                           quantization_configurations=quantization_configurations)

        tpc = generate_custom_test_tpc(
            name="custom_16_bit_tpc",
            base_cfg=tpc.default_qco.base_config,
            base_tpc=tpc,
            operator_sets_dict={
                OperatorSetNames.MUL: qco_16,
            })

        return tpc

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(activation_memory=5000)

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig()

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.multiply(inputs, inputs)[:, :8, :8, :]
        x = tf.add(x, np.ones((3,), dtype=np.float32))
        x1 = tf.subtract(x, np.ones((3,), dtype=np.float32))
        x = tf.multiply(x, x1)
        x = tf.keras.layers.Conv2D(3, 1)(x)
        outputs = tf.divide(x, 2*np.ones((3,), dtype=np.float32))
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        act_quant_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        mul1_act_quant, mul2_act_quant = act_quant_layers[1], act_quant_layers[4]
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "1st mul activation bits should be 8 bits because of RU.")
        self.unit_test.assertTrue(mul1_act_quant.activation_holder_quantizer.signed == False,
                                  "1st mul activation should be unsigned, because activations as all positive.")
        self.unit_test.assertTrue(mul2_act_quant.activation_holder_quantizer.num_bits == 8,
                                  "2nd mul activation bits should be 8 bits because of following div node.")
