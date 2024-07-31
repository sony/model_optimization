# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from keras.activations import sigmoid, softmax

from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.core.keras.constants import SIGMOID, SOFTMAX
from tests.common_tests.helpers.generate_test_tp_model import generate_test_op_qc, generate_test_attr_configs
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from keras import backend as K

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class MixedPrecisionActivationBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_layers_idx, num_calibration_iter=1):
        super().__init__(unit_test, num_calibration_iter=num_calibration_iter)

        self.activation_layers_idx = activation_layers_idx

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                           mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False,
                                           weights_bias_correction=True,
                                           input_scaling=False,
                                           activation_channel_equalization=False)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 4)(x)
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def verify_quantization(self, quantized_model, input_x, weights_layers_idx, weights_layers_channels_size,
                            activation_layers_idx, unique_tensor_values):
        # verify weights quantization
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        for conv_layer, num_channels in zip(conv_layers,weights_layers_channels_size):
            for j in range(num_channels):  # quantized per channel
                self.unit_test.assertTrue(
                    np.unique(conv_layer.get_quantized_weights()['kernel'][:, :, :, j]).flatten().shape[
                        0] <= unique_tensor_values)

        # verify activation quantization
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:] # skip the input layer
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in holder_layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])

        # verifying fake quant nodes output
        for layer_out in layer_outs:
            self.unit_test.assertTrue(np.unique(layer_out).flatten().shape[0] <= unique_tensor_values)


class MixedPrecisionActivationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=17919, activation_memory=5407)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationSearch4BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2,4])

    def get_resource_utilization(self):
        # resource utilization is for 4 bits on average
        return ResourceUtilization(weights_memory=17920 * 4 / 8, activation_memory=5408 * 4 / 8)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is 4 bit average
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]

        # Note that since we're using default max aggregation for activation resource utilization,
        # then there is no guarantee that the activation bitwidth for each layer would be 4-bit,
        # this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the input layer's bitwidth)
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        # Verify final resource utilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionActivationSearch2BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        # resource utilization is for 2 bits on average
        return ResourceUtilization(weights_memory=17920.0 * 2 / 8, activation_memory=5408.0 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is minimal
        # Note that since we're using default max aggregation for activation resource utilization, then there is no guarantee that the
        # activation bitwidth for each layer would be 2-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the input layer's bitwidth)
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=4)

        # Verify final resource utilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionActivationDepthwiseTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3])

    def get_resource_utilization(self):
        return ResourceUtilization(47, 767)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 8]))


class MixedPrecisionActivationDepthwise4BitTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1])

    def get_resource_utilization(self):
        return ResourceUtilization(48.0 * 4 / 8, 768.0 * 4 / 8)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_depthwise_4bit_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is 4 bit average
        # Note that since we're using default max aggregation for activation resource utilization, then there is no guarantee that the
        # activation bitwidth for each layer would be 4-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the relu layer's bitwidth)
        holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[0]
        self.unit_test.assertTrue(holder_layer.activation_holder_quantizer.get_config()['num_bits']==4)


class MixedPrecisionActivationSplitLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 4])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.split(inputs, num_or_size_splits=2, axis=1)
        c0 = layers.Conv2D(32, 4)(x[0])
        c1 = layers.Conv2D(32, 4)(x[1])
        model = keras.Model(inputs=inputs, outputs=[c0, c1])
        return model

    def get_resource_utilization(self):
        return ResourceUtilization(3071, 2079)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[3, 4],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationOnlyTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 4])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        outputs = layers.Conv2D(32, 4)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_weights_disabled_test")

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 5407)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.activation_memory + quantization_info.final_resource_utilization.weights_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running activation mixed-precision with unconstrained weights and total resource utilization, "
            "final total memory should be equal to the sum of activation and weights memory.")


class MixedPrecisionActivationOnlyWeightsDisabledTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 3])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Conv2D(32, 4)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(enable_kernel_weights_quantization=False))
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})

        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_weights_disabled_test")

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 5407)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationAddLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 3])

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 5407)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Add()([x, x])
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [h.activation_holder_quantizer.get_config()['num_bits'] for h in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2],
                                 weights_layers_channels_size=[32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationMultipleInputsTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=3, activation_layers_idx=[4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.num_of_inputs = 4
        self.val_batch_size = 2

    def get_resource_utilization(self):
        return ResourceUtilization(6143, 6817408)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3] for _ in range(self.num_of_inputs)]

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                           input_scaling=False, activation_channel_equalization=False)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=self.num_of_inputs)

    def create_networks(self):
        inputs_1 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_2 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_3 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_4 = layers.Input(shape=self.get_input_shapes()[0][1:])
        x1 = layers.Conv2D(32, 4)(inputs_1)
        x2 = layers.Conv2D(32, 4)(inputs_2)
        x3 = layers.Conv2D(32, 4)(inputs_3)
        x4 = layers.Conv2D(32, 4)(inputs_4)
        outputs = layers.Concatenate()([x1, x2, x3, x4])
        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3, inputs_4], outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8, 8, 8, 8, 8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionTotalMemoryUtilizationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, np.inf, total_memory=(17920 + 5408) * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionMultipleResourcesTightUtilizationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        weights = 17920 * 4 / 8
        activation = 5408 * 4 / 8
        return ResourceUtilization(weights, activation, total_memory=weights + activation)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionReducedTotalMemorySearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        weights = 17920 * 4 / 8
        activation = 5408 * 4 / 8
        return ResourceUtilization(weights, activation, total_memory=(weights + activation) / 2)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionDistanceSoftmaxTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 767)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})

        custom_opsets = {"Softmax": [layers.Softmax, tf.nn.softmax, softmax,
                                     tp.LayerFilterParams(layers.Activation, activation=SOFTMAX)]}
        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_test",
                                                custom_opsets=custom_opsets)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Softmax()(inputs)
        x = tf.nn.softmax(x)
        x = softmax(x)
        x = layers.Activation(SOFTMAX)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4, 4, 4, 4]))


class MixedPrecisionDistanceSigmoidTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 767)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = sigmoid(inputs)
        x = tf.nn.sigmoid(x)
        x = layers.Activation(SIGMOID)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in
                           holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4, 4, 4]))
