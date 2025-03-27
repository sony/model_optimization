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
from mct_quantizers import KerasActivationQuantizationHolder, KerasQuantizationWrapper
from model_compression_toolkit.core.common.network_editors import NodeNameFilter, NodeTypeFilter
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, KERAS_KERNEL, BIAS
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from tests.common_tests.helpers.generate_test_tpc import generate_test_op_qc, generate_test_attr_configs, \
    generate_custom_test_tpc
from tests.common_tests.helpers.tpcs_for_tests.v3.tpc import get_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras

#from model_compression_toolkit.core.keras.constants import KERNEL, BIAS

keras = tf.keras
layers = keras.layers

get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class ManualWeightsBitWidthSelectionTest(BaseKerasFeatureNetworkTest):
    """
    This test check the manual weights bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths,
    attribute or list of attribute.
    Uses the manual bit width API in the "get_core_configs" method.
    """

    def __init__(self, unit_test, filters, bit_widths, attrs, **kwargs):
        self.filters = filters
        self.bit_widths = bit_widths
        self.attrs = attrs
        self.layer_types = []
        self.layer_names = []

        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        attrs = [attrs] if not isinstance(attrs, list) else attrs
        if len(bit_widths) < len(filters):
            bit_widths = [bit_widths[0] for f in filters]
        if len(attrs) < len(filters):
            attrs = [attrs[0] for f in attrs]
        for filter, bit_width, attr in zip(filters, bit_widths, attrs):
            if isinstance(filter, NodeNameFilter):
                self.layer_names.append([filter.node_name, [bit_width, attr]])
            elif isinstance(filter, NodeTypeFilter):
                self.layer_types.append([filter.node_type, [bit_width, attr]])
        super().__init__(unit_test, **kwargs)

    def create_networks(self):
        input_tensor = layers.Input(shape=self.get_input_shapes()[0][1:], name='input')
        x1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='conv1')(input_tensor)
        x1 = layers.Add(name='add1')([x1, np.ones((3,), dtype=np.float32)])

        # Second convolutional block
        x2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='conv2')(x1)
        x2 = layers.BatchNormalization(name='bn1')(x2)
        x2 = layers.ReLU(name='relu1')(x2)

        # Addition
        x = layers.Add(name='add2')([x1, x2])

        # Flatten and fully connected layer
        x = layers.Flatten()(x)
        output_tensor = layers.Dense(units=10, activation='softmax', name='fc')(x)

        return keras.Model(inputs=input_tensor, outputs=output_tensor)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(kernel_cfg_nbits=16, bias_cfg_nbits=16, enable_bias_weights_quantization=True))

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set 16, 8, 4 and 2 bit candidates for test
        mixed_precision_candidates_list = [(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")

    def get_mp_core_config(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False,
                                         custom_tpc_opset_to_layer={'Input': CustomOpsetLayers([layers.InputLayer])})
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        core_config = mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)
        return core_config

    def get_core_config(self):
        # Configures the core settings including manual bit width adjustments.
        core_config = self.get_mp_core_config()
        core_config.bit_width_config.set_manual_weights_bit_width(self.filters, self.bit_widths, self.attrs)
        return core_config

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # in the compare we need bit_widths to be a list
        bit_widths = [self.bit_widths] if not isinstance(self.bit_widths, list) else self.bit_widths
        attrs = [self.attrs] if not isinstance(self.attrs, list) else self.attrs

        for layer in quantized_model.layers:
            # check if the layer is an weights quantizer
            if isinstance(layer, KerasQuantizationWrapper):
                ## get the layer that's weights is being quantized
                layer_q = quantized_model.layers[quantized_model.layers.index(layer)]
                if isinstance(layer_q, KerasQuantizationWrapper):
                    layer_q = layer_q.layer

                # check if this layer is in the layer types to change bit width and check that the correct bit width was applied.
                layer_q_bitwidth_attrs = []
                for layer_name in self.layer_names:
                    if layer_name[0] == layer_q.name:
                        layer_q_bitwidth_attrs.append(layer_name[1])
                for layer_type in self.layer_types:
                    if layer_type[0] == type(layer_q):
                        layer_q_bitwidth_attrs.append(layer_type[1])

                for layer_q_bitwidth_attr in layer_q_bitwidth_attrs:
                    layer_q_bit_width = layer_q_bitwidth_attr[0]
                    layer_q_attr      = layer_q_bitwidth_attr[1]

                    if layer.weights_quantizers.get(layer_q_attr) is not None:
                        if layer_q_bit_width is not None:
                            self.unit_test.assertTrue(layer.weights_quantizers.get(layer_q_attr).num_bits == layer_q_bit_width)
                    else:
                        # make sure that the bit width of other layers was not changed.
                        self.unit_test.assertFalse(layer.weights_quantizers.get(layer_q_attr).num_bits in bit_widths,
                                                   msg=f"name {layer_q.name}, layer.KerasQuantizationWrapper.num_bits {layer.weights_quantizers.get(layer_q_attr).num_bits}, {self.bit_widths}")


class ManualWeightsBias2BitWidthSelectionTest(ManualWeightsBitWidthSelectionTest):
    """
    This test check the manual bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(kernel_cfg_nbits=8, bias_cfg_nbits=2, enable_bias_weights_quantization=True))

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set 16, 8, 4 and 2 bit candidates for test
        mixed_precision_candidates_list = [(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")


class ManualWeightsBias4BitWidthSelectionTest(ManualWeightsBitWidthSelectionTest):
    """
    This test check the manual bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(kernel_cfg_nbits=8, bias_cfg_nbits=4, enable_bias_weights_quantization=True))

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set 16, 8, 4 and 2 bit candidates for test
        mixed_precision_candidates_list = [(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")


class ManualWeightsBias8BitWidthSelectionTest(ManualWeightsBitWidthSelectionTest):
    """
    This test check the manual bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(kernel_cfg_nbits=8, bias_cfg_nbits=8, enable_bias_weights_quantization=True))

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set 16, 8, 4 and 2 bit candidates for test
        mixed_precision_candidates_list = [(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")


class ManualWeightsBias32BitWidthSelectionTest(ManualWeightsBitWidthSelectionTest):
    """
    This test check the manual bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(kernel_cfg_nbits=8, bias_cfg_nbits=32, enable_bias_weights_quantization=True))

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set 16, 8, 4 and 2 bit candidates for test
        mixed_precision_candidates_list = [(8, 8), (4, 8), (2, 8), (16, 8), (8, 16), (4, 16), (2, 16), (16, 16)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")
