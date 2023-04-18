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
from keras.layers import DepthwiseConv2D, ReLU

from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_last_layer_weights
from model_compression_toolkit.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class MixedPercisionBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, val_batch_size=1):
        super().__init__(unit_test, val_batch_size=val_batch_size, experimental_exporter=True)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                      mct.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=True,
                                      weights_bias_correction=True,
                                      weights_per_channel_threshold=True,
                                      input_scaling=True,
                                      activation_channel_equalization=True)

    def get_mixed_precision_v2_config(self):
        return mct.MixedPrecisionQuantizationConfigV2(num_of_images=1)

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


class MixedPercisionManuallyConfiguredTest(MixedPercisionBaseTest):

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             mp_bitwidth_candidates_list=[(8, 8), (2, 8), (3, 8)],
                                             name="mp_test")

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE, mct.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=True, weights_bias_correction=True,
                                      weights_per_channel_threshold=False, input_scaling=True,
                                      activation_channel_equalization=True)

    def get_mixed_precision_v2_config(self):
        return mct.MixedPrecisionQuantizationConfigV2()

    def get_kpi(self):
        # Return some KPI (it does not really matter the value here as search_methods is not done,
        # and the configuration is
        # set manually)
        return KPI(1)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert quantization_info.mixed_precision_cfg == [2, 1]
        self.unit_test.assertTrue(np.unique(quantized_model.layers[2].weights[0]).flatten().shape[0] <= 4)
        self.unit_test.assertTrue(np.unique(quantized_model.layers[4].weights[0]).flatten().shape[0] <= 8)


class MixedPercisionSearchTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[3].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPercisionSearchKPI4BitsAvgTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 4 bits on average
        return KPI(17920 * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [1, 1]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[3].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPercisionSearchKPI2BitsAvgTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(17920 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [2, 2]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 4)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[3].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 4)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPercisionSearchActivationKPINonConfNodesTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        # Total KPI for weights in 2 bit avg and non-configurable activation in 8 bit
        self.target_total_kpi = KPI(weights_memory=17920 * 2 / 8, activation_memory=5408)

    def get_kpi(self):
        return self.target_total_kpi

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # No need to verify quantization configuration here since this test is similar to other tests we have,
        # we're only interested in the KPI
        self.unit_test.assertTrue(quantization_info.final_kpi.activation_memory <=
                                  self.target_total_kpi.activation_memory)
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPercisionSearchTotalKPINonConfNodesTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        # Total KPI for weights in 2 bit avg and non-configurable activation in 8 bit
        self.target_total_kpi = KPI(total_memory=17920 * 2 / 8 + 5408)

    def get_kpi(self):
        return self.target_total_kpi

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # No need to verify quantization configuration here since this test is similar to other tests we have,
        # we're only interested in the KPI
        self.unit_test.assertTrue(quantization_info.final_kpi.total_memory <= self.target_total_kpi.total_memory)
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")



class MixedPercisionDepthwiseTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].layer, DepthwiseConv2D))
        self.unit_test.assertTrue(len(quantization_info.mixed_precision_cfg) == 1)
        self.unit_test.assertTrue(quantization_info.mixed_precision_cfg[0] == 0) # Assert model is quantized using 16 bits as KPI is inf
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].layer, ReLU))

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(weights_n_bits=16,
                                                 activation_n_bits=16)

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             mp_bitwidth_candidates_list=[(8, 16), (2, 16), (4, 16), (16, 16)],
                                             name="mp_dw_test")

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                      mct.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=False,
                                      weights_bias_correction=False,
                                      weights_per_channel_threshold=True,
                                      input_scaling=False,
                                      activation_channel_equalization=False)

    def get_mixed_precision_v2_config(self):
        return mct.MixedPrecisionQuantizationConfigV2()


class MixedPrecisionActivationDisabled(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                      mct.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=True,
                                      weights_bias_correction=True,
                                      weights_per_channel_threshold=True,
                                      input_scaling=False,
                                      activation_channel_equalization=False)

    def get_mixed_precision_v2_config(self):
        return mct.MixedPrecisionQuantizationConfigV2(num_of_images=1)

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        activation_disabled_config = base_config.clone_and_edit(enable_activation_quantization=False)

        return get_weights_only_mp_tpc_keras(base_config=activation_disabled_config,
                                             mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8)],
                                             name="mp_weights_only_test")

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[3].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)


class MixedPercisionSearchLastLayerDistanceTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_mixed_precision_v2_config(self):
        return mct.MixedPrecisionQuantizationConfigV2(num_of_images=1,
                                                      distance_weighting_method=get_last_layer_weights,
                                                      use_grad_based_weights=False)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[3].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")