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

from mct_quantizers import KerasQuantizationWrapper
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_KERNEL, BIAS_ATTR, BIAS
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs, generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_op_qc, generate_test_attr_configs
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class MixedPrecisionBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, val_batch_size=1, num_calibration_iter=1):
        super().__init__(unit_test, val_batch_size=val_batch_size, num_calibration_iter=num_calibration_iter)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=True, weights_bias_correction=True,
                                           input_scaling=True, activation_channel_equalization=True)

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


class MixedPrecisionManuallyConfiguredTest(MixedPrecisionBaseTest):

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             default_config=default_config,
                                             mp_bitwidth_candidates_list=[(8, 8), (2, 8), (3, 8)],
                                             name="mp_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=True, weights_bias_correction=True,
                                           input_scaling=True, activation_channel_equalization=True)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(target_resource_utilization=self.get_resource_utilization())

    def get_resource_utilization(self):
        # Return some ResourceUtilization (it does not really matter the value here as search_methods is not done,
        # and the configuration is
        # set manually)
        return ResourceUtilization(1)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert quantization_info.mixed_precision_cfg == [2, 1]
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(np.unique(conv_layers[0].weights[0]).flatten().shape[0] <= 4)
        self.unit_test.assertTrue(np.unique(conv_layers[1].weights[0]).flatten().shape[0] <= 8)


class MixedPrecisionSearchTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test, distance_metric=MpDistanceWeighting.AVG):
        super().__init__(unit_test, val_batch_size=2)
        self.distance_metric = distance_metric

    def get_resource_utilization(self):
        return ResourceUtilization(17919)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                         distance_weighting_method=self.distance_metric)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(any([b != 0 for b in quantization_info.mixed_precision_cfg]),
                                  "At least one of the conv layers is expected to be quantized to meet the required "
                                  "resource utilization target.")
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained ResourceUtilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionWithHessianScoresTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test, distance_metric=MpDistanceWeighting.AVG):
        super().__init__(unit_test, val_batch_size=2, num_calibration_iter=10)
        self.distance_metric = distance_metric

    def get_resource_utilization(self):
        return ResourceUtilization(17919)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=10,
                                                         distance_weighting_method=self.distance_metric,
                                                         use_hessian_based_scores=True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(any([b != 0 for b in quantization_info.mixed_precision_cfg]),
                                  "At least one of the conv layers is expected to be quantized to meet the required "
                                  "resource utilization target.")
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained ResourceUtilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionSearchPartWeightsLayersTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_tpc(self):
        # Building a TPC that gives Conv layers mixed precision candidates and Dense layers a fixed candidate.
        # Both layers that have weights to quantized, so we want to verify that finalizing the model is successful.
        # Note that this is important that the quantization config options would include also activation quantization.
        cfg, mixed_precision_cfg_list, _ = get_op_quantization_configs()

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
            tp.OperatorsSet("Weights_mp", weight_mixed_cfg)
            tp.OperatorsSet("Weights_fixed", weight_fixed_cfg)

        keras_tpc = tp.TargetPlatformCapabilities(tp_model, name="mp_part_weights_layers_test")

        with keras_tpc:
            tp.OperationsSetToLayers(
                "Weights_fixed",
                [layers.Dense],
                attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                              BIAS_ATTR: DefaultDict(default_value=BIAS)}
            )

            tp.OperationsSetToLayers(
                "Weights_mp",
                [layers.Conv2D],
                attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                              BIAS_ATTR: DefaultDict(default_value=BIAS)}
            )

        return keras_tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Dense(32)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def get_resource_utilization(self):
        return ResourceUtilization(1790)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # We just needed to verify that the graph finalization is working without failing.
        # The actual quantization is not interesting for the sake of this test, so we just verify some
        # degenerated things to see that everything worked.
        self.unit_test.assertTrue(quantization_info.mixed_precision_cfg == [1])

        dense_layer = get_layers_from_model_by_type(quantized_model, layers.Dense)
        self.unit_test.assertTrue(len(dense_layer) == 1)
        dense_layer = dense_layer[0]
        for i in range(32):  # quantized to 2 bits per channel
            self.unit_test.assertTrue(
                np.unique(dense_layer.get_quantized_weights()['kernel'][:, i]).flatten().shape[0] <= 4)


class MixedPrecisionSearch4BitsAvgTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        # Resource Utilization is for 4 bits on average
        return ResourceUtilization(17920 * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [1, 1]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained ResourceUtilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionCombinedNMSTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                         use_hessian_based_scores=False)

    def get_resource_utilization(self):
        # Resource Utilization is for 4 bits on average
        return ResourceUtilization(17920 * 4 / 8)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 4)(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((160, 5, 4))(x)
        outputs = tf.image.combined_non_max_suppression(x, tf.reduce_mean(x, 3), 10, 10)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue((quantization_info.mixed_precision_cfg != 0).any())

        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16 or
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained ResourceUtilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionSearch2BitsAvgTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        # Resource Utilization is for 2 bits on average
        return ResourceUtilization(17920 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [2, 2]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 4)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 4)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained ResourceUtilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionSearchActivationNonConfNodesTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        # Total ResourceUtilization for weights in 2 bit avg and non-configurable activation in 8 bit
        self.target_total_ru = ResourceUtilization(weights_memory=17920 * 2 / 8, activation_memory=5408)

    def get_resource_utilization(self):
        return self.target_total_ru

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # No need to verify quantization configuration here since this test is similar to other tests we have,
        # we're only interested in the ResourceUtilization
        self.unit_test.assertTrue(quantization_info.final_resource_utilization.activation_memory <=
                                  self.target_total_ru.activation_memory)
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained Resource Utilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionSearchTotalMemoryNonConfNodesTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        # Total ResourceUtilization for weights in 2 bit avg and non-configurable activation in 8 bit
        self.target_total_ru = ResourceUtilization(total_memory=17920 * 2 / 8 + 5408)

    def get_resource_utilization(self):
        return self.target_total_ru

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # No need to verify quantization configuration here since this test is similar to other tests we have,
        # we're only interested in the ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory <= self.target_total_ru.total_memory)
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained ResourceUtilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionDepthwiseTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(95)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantization_info.mixed_precision_cfg) == 1)
        self.unit_test.assertTrue(quantization_info.mixed_precision_cfg[0] == 1)

    def get_tpc(self):
        base_config = generate_test_op_qc(activation_n_bits=16,
                                          **generate_test_attr_configs(default_cfg_nbits=16,
                                                                       kernel_cfg_nbits=16))

        default_config = base_config.clone_and_edit(attr_weights_configs_mapping={})

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             default_config=default_config,
                                             mp_bitwidth_candidates_list=[(8, 16), (2, 16), (4, 16), (16, 16)],
                                             name="mp_dw_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False, weights_bias_correction=False,
                                           input_scaling=False, activation_channel_equalization=False)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig()


class MixedPrecisionActivationDisabled(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                           mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=True,
                                           weights_bias_correction=True,
                                           input_scaling=False,
                                           activation_channel_equalization=False)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        activation_disabled_config = base_config.clone_and_edit(enable_activation_quantization=False)

        return get_weights_only_mp_tpc_keras(base_config=activation_disabled_config,
                                             default_config=default_config,
                                             mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8)],
                                             name="mp_weights_only_test")

    def get_resource_utilization(self):
        # resource utilization is infinity -> should give best model - 8bits
        return ResourceUtilization(17919)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert (quantization_info.mixed_precision_cfg == [0, 1]).all()
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)


class MixedPrecisionSearchLastLayerDistanceTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, val_batch_size=2)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                         distance_weighting_method=MpDistanceWeighting.LAST_LAYER,
                                                         use_hessian_based_scores=False)

    def get_resource_utilization(self):
        return ResourceUtilization(17919)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        assert any([(quantization_info.mixed_precision_cfg == [1, 0]).all(),
                    (quantization_info.mixed_precision_cfg == [0, 1]).all()])
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[0].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(32):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(conv_layers[1].get_quantized_weights()['kernel'][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final Resource Utilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running weights mixed-precision with unconstrained Resource Utilization, "
            "final weights and activation memory sum should be equal to total memory.")


class MixedPrecisionWeightsOnlyConfigurableActivationsTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Add()([x, x])
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        cfg, mixed_precision_cfg_list, _ = get_op_quantization_configs()

        act_eight_bit_cfg = cfg.clone_and_edit(activation_n_bits=8,
                                               attr_weights_configs_mapping={})
        act_four_bit_cfg = cfg.clone_and_edit(activation_n_bits=4,
                                              attr_weights_configs_mapping={})
        act_two_bit_cfg = cfg.clone_and_edit(activation_n_bits=2,
                                             attr_weights_configs_mapping={})

        mixed_precision_cfg_list = \
            [c.clone_and_edit(enable_activation_quantization=False) for c in mixed_precision_cfg_list]
        cfg = mixed_precision_cfg_list[0]

        act_mixed_cfg = tp.QuantizationConfigOptions(
            [act_eight_bit_cfg, act_four_bit_cfg, act_two_bit_cfg],
            base_config=act_eight_bit_cfg,
        )

        weight_mixed_cfg = tp.QuantizationConfigOptions(
            mixed_precision_cfg_list,
            base_config=cfg,
        )

        tp_model = tp.TargetPlatformModel(tp.QuantizationConfigOptions([cfg], cfg),
                                          name="mp_weights_conf_act_test")

        with tp_model:
            tp.OperatorsSet("Activations", act_mixed_cfg)
            tp.OperatorsSet("Weights", weight_mixed_cfg)

        keras_tpc = tp.TargetPlatformCapabilities(tp_model, name="mp_weights_conf_act_test")

        with keras_tpc:
            tp.OperationsSetToLayers(
                "Weights",
                [layers.Conv2D],
                attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                              BIAS_ATTR: DefaultDict(default_value=BIAS)}
            )

            tp.OperationsSetToLayers(
                "Activations",
                [layers.ReLU, layers.Add]
            )

        return keras_tpc

    def get_resource_utilization(self):
        return ResourceUtilization(1535)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        wrapper_layers = get_layers_from_model_by_type(quantized_model, KerasQuantizationWrapper)
        weights_bits = wrapper_layers[0].weights_quantizers[KERNEL].num_bits
        self.unit_test.assertTrue(weights_bits == 4)
