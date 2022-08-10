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
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from keras import backend as K

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.tpc_keras import generate_activation_mp_tpc_keras

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def get_base_eight_bits_config_op():
    return tp.OpQuantizationConfig(
            activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
            weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True,
            enable_activation_quantization=True,
            quantization_preserving=False,
            fixed_scale=None,
            fixed_zero_point=None,
            weights_multiplier_nbits=None
        )


def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class MixedPrecisionActivationBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_layers_idx):
        super().__init__(unit_test)

        self.activation_layers_idx = activation_layers_idx

    def get_tpc(self):
        eight_bits = get_base_eight_bits_config_op()

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        tp = generate_tp_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_tp')
        return generate_activation_mp_tpc_keras(name="mixed_precision_activation_test", tp_model=tp)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=False,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    input_scaling=False,
                                    activation_channel_equalization=False)

        return MixedPrecisionQuantizationConfig(qc, num_of_images=1)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(50, 40)(x)
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
        for i in range(len(weights_layers_idx)):
            for j in range(weights_layers_channels_size[i]):  # quantized per channel
                self.unit_test.assertTrue(
                    np.unique(quantized_model.layers[weights_layers_idx[i]].weights[0][:, :, :, j]).flatten().shape[
                        0] <= unique_tensor_values)

        # verify activation quantization
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in quantized_model.layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])
        # verifying fake quant nodes output
        for idx in activation_layers_idx:
            self.unit_test.assertTrue(np.unique(layer_outs[idx].flatten()).shape[0] <= unique_tensor_values)


class MixedPrecisionActivationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 7])

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationSearchKPI4BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 5])

    def get_kpi(self):
        # kpi is for 4 bits on average
        return KPI(weights_memory=2544000 * 4 / 8, activation_memory=1211800 * 4 / 8)

    def get_tpc(self):
        eight_bits = get_base_eight_bits_config_op()
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        tp = generate_tp_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_tp')
        return generate_activation_mp_tpc_keras(name="mixed_precision_4bit_test", tp_model=tp)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is 4 bit average
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        # Note that since we're using default max aggregation for activation KPI, then there is no guarantee that the
        # activation bitwidth for each layer would be 4-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the input layer's bitwidth)
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.total_memory ==
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionActivationSearchKPI2BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 5])

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(weights_memory=2544000.0 * 2 / 8, activation_memory=1211800.0 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is minimal
        # Note that since we're using default max aggregation for activation KPI, then there is no guarantee that the
        # activation bitwidth for each layer would be 2-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the input layer's bitwidth)
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=4)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.total_memory ==
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionActivationDepthwiseTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 5])

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(30)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8]))

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        # quantifying both weights and activation so similarity is approximately within error range of 1e-3
        self.unit_test.assertTrue(np.isclose(cs, 1, rtol=1e-3), msg=f'fail cosine similarity check:{cs}')


class MixedPrecisionActivationDepthwise4BitTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1])

    def get_kpi(self):
        # return KPI(np.inf, np.inf)
        return KPI(2700.0 * 4 / 8, 289743.0 * 4 / 8)

    def get_tpc(self):
        eight_bits = get_base_eight_bits_config_op()
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        tp = generate_tp_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_tp')
        return generate_activation_mp_tpc_keras(name="mixed_precision_depthwise_4bit_test", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(30)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is 4 bit average
        # Note that since we're using default max aggregation for activation KPI, then there is no guarantee that the
        # activation bitwidth for each layer would be 4-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the relu layer's bitwidth)
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [4]))


class MixedPrecisionActivationSplitLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 5, 6])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.split(inputs, num_or_size_splits=2, axis=1)
        c0 = layers.Conv2D(30, 40)(x[0])
        c1 = layers.Conv2D(30, 40)(x[1])
        model = keras.Model(inputs=inputs, outputs=[c0, c1])
        return model

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[3, 4],
                                 weights_layers_channels_size=[30, 30],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationOnlyTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 5])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Conv2D(30, 40)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        eight_bits = get_base_eight_bits_config_op()
        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        tp = generate_tp_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_tp')
        return generate_activation_mp_tpc_keras(name="mixed_precision_activation_weights_disabled_test", tp_model=tp)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.activation_memory == quantization_info.final_kpi.total_memory,
            "Running activation mixed-precision with unconstrained weights and total KPI, "
            "final activation memory and total memory should be equal.")
        self.unit_test.assertTrue(quantization_info.final_kpi.weights_memory == 0,
                                  "Running activation only mixed-precision, final weights memory should be 0.")


class MixedPrecisionActivationOnlyWeightsDisabledTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 5])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Conv2D(30, 40)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        eight_bits = get_base_eight_bits_config_op()
        weights_disabled_config = eight_bits.clone_and_edit(enable_weights_quantization=False)

        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        tp = generate_tp_model_with_activation_mp(weights_disabled_config, mixed_precision_candidates_list, name='mp_default_tp')
        return generate_activation_mp_tpc_keras(name="mixed_precision_activation_weights_disabled_test", tp_model=tp)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationAddLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 5])

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.Add()([x, x])
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2],
                                 weights_layers_channels_size=[30],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationMultipleInputsTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 4, 5, 9, 10, 11])
        self.num_of_inputs = 3

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3] for _ in range(self.num_of_inputs)]

    def create_networks(self):
        inputs_1 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_2 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_3 = layers.Input(shape=self.get_input_shapes()[0][1:])
        x1 = layers.Conv2D(30, 40)(inputs_1)
        x2 = layers.Conv2D(30, 40)(inputs_2)
        x3 = layers.Conv2D(30, 40)(inputs_3)
        outputs = layers.Concatenate()([x1, x2, x3])
        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # kpi is infinity -> should give best model - 8bits
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8, 8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionTotalKPISearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 5])

    def get_kpi(self):
        return KPI(np.inf, np.inf, total_memory=(2544000 + 1211800) * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in
                           self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.total_memory ==
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionMultipleKPIsTightSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 5])

    def get_kpi(self):
        weights = 2544000 * 4 / 8
        activation = 1211800 * 4 / 8
        return KPI(weights, activation, total_memory=weights + activation)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in
                           self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.total_memory ==
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionReducedTotalKPISearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 5])

    def get_kpi(self):
        weights = 2544000 * 4 / 8
        activation = 1211800 * 4 / 8
        return KPI(weights, activation, total_memory=(weights + activation) / 2)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        activation_bits = [quantized_model.layers[i].inbound_nodes[0].call_kwargs.get('num_bits') for i in
                           self.activation_layers_idx]
        self.unit_test.assertTrue((activation_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.total_memory ==
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")
