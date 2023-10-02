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

from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_last_layer_weights
from model_compression_toolkit.target_platform_capabilities.target_platform.quantization_format import \
    QuantizationFormat
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class MixedPercisionBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, val_batch_size=1):
        super().__init__(unit_test, val_batch_size=val_batch_size, experimental_exporter=True)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=True,
                                      weights_bias_correction=True,
                                      weights_per_channel_threshold=True,
                                      input_scaling=True,
                                      activation_channel_equalization=True)

    def get_mixed_precision_v2_config(self):
        return mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1)

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
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=True, weights_bias_correction=True,
                                      weights_per_channel_threshold=False, input_scaling=True,
                                      activation_channel_equalization=True)

    def get_mixed_precision_v2_config(self):
        return mct.core.MixedPrecisionQuantizationConfigV2()

    def get_kpi(self):
        # Return some KPI (it does not really matter the value here as search_methods is not done,
        # and the configuration is
        # set manually)
        return KPI(1)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert quantization_info.mixed_precision_cfg == [2, 1]
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(np.unique(conv_layers[0].weights[0]).flatten().shape[0] <= 4)
        self.unit_test.assertTrue(np.unique(conv_layers[1].weights[0]).flatten().shape[0] <= 8)


class MixedPercisionSearchTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPercisionSearchPartWeightsLayersTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_tpc(self):
        # Building a TPC that gives Conv layers mixed precision candidates and Dense layers a fixed candidate.
        # Both layers that have weights to quantized, so we want to verify that finalizing the model is successful.
        # Note that this is important that the quantization config options would include also activation quantization.
        cfg, mixed_precision_cfg_list = get_op_quantization_configs()

        two_bit_cfg = mixed_precision_cfg_list[2]

        weight_mixed_cfg = tp.QuantizationConfigOptions(
            mixed_precision_cfg_list,
            base_config=cfg,
        )

        weight_fixed_cfg = tp.QuantizationConfigOptions(
            [two_bit_cfg],
            base_config=two_bit_cfg,
        )

        tp_model = tp.TargetPlatformModel(weight_fixed_cfg, name="mp_part_weights_layers_test")
        with tp_model:
            tp_model.set_quantization_format(QuantizationFormat.FAKELY_QUANT)

            tp.OperatorsSet("Weights_mp", weight_mixed_cfg)
            tp.OperatorsSet("Weights_fixed", weight_fixed_cfg)

        keras_tpc = tp.TargetPlatformCapabilities(tp_model, name="mp_part_weights_layers_test")

        with keras_tpc:
            tp.OperationsSetToLayers(
                "Weights_fixed",
                [layers.Dense],
            )

            tp.OperationsSetToLayers(
                "Weights_mp",
                [layers.Conv2D],
            )

        return keras_tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Dense(32)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # We just needed to verify that the graph finalization is working without failing.
        # The actual quantization is not interesting for the sake of this test, so we just verify some
        # degenerated things to see that everything worked.
        self.unit_test.assertTrue(quantization_info.mixed_precision_cfg == [0])  # kpi is infinity -> should give best model - 8bits

        dense_layer = get_layers_from_model_by_type(quantized_model, layers.Dense)
        self.unit_test.assertTrue(len(dense_layer) == 1)
        dense_layer = dense_layer[0]
        for i in range(32):  # quantized to 2 bits per channel
            self.unit_test.assertTrue(
                np.unique(dense_layer.get_quantized_weights()['kernel'][:, i]).flatten().shape[0] <= 4)


class MixedPercisionSearchKPI4BitsAvgTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 4 bits on average
        return KPI(17920 * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [1, 1]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)

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
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [2, 2]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 4)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 4)

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
        self.unit_test.assertTrue(len(quantization_info.mixed_precision_cfg) == 1)
        self.unit_test.assertTrue(quantization_info.mixed_precision_cfg[0] == 0) # Assert model is quantized using 16 bits as KPI is inf


    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(weights_n_bits=16,
                                                 activation_n_bits=16)

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             mp_bitwidth_candidates_list=[(8, 16), (2, 16), (4, 16), (16, 16)],
                                             name="mp_dw_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=False,
                                      weights_bias_correction=False,
                                      weights_per_channel_threshold=True,
                                      input_scaling=False,
                                      activation_channel_equalization=False)

    def get_mixed_precision_v2_config(self):
        return mct.core.MixedPrecisionQuantizationConfigV2()


class MixedPrecisionActivationDisabled(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                      mct.core.QuantizationErrorMethod.MSE,
                                      relu_bound_to_power_of_2=True,
                                      weights_bias_correction=True,
                                      weights_per_channel_threshold=True,
                                      input_scaling=False,
                                      activation_channel_equalization=False)

    def get_mixed_precision_v2_config(self):
        return mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1)

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
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)


class MixedPercisionSearchLastLayerDistanceTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_mixed_precision_v2_config(self):
        return mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1,
                                                      distance_weighting_method=get_last_layer_weights,
                                                      use_grad_based_weights=False)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory + quantization_info.final_kpi.activation_memory ==
            quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights and activation memory sum should be equal to total memory.")