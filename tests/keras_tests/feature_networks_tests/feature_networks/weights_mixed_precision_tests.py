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

from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class MixedPercisionBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_mixed_precision_v2_config(self):
        return MixedPrecisionQuantizationConfigV2(num_of_images=1)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=True,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    input_scaling=True,
                                    activation_channel_equalization=True)

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


class MixedPercisionManuallyConfiguredTest(MixedPercisionBaseTest):

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        mp_tp_model = generate_mixed_precision_test_tp_model(base_cfg=base_config,
                                                             mp_bitwidth_candidates_list=[(8, 8), (2, 8), (3, 8)])
        return generate_keras_tpc(name="mp_test", tp_model=mp_tp_model)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE, mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=True, weights_bias_correction=True,
                                    weights_per_channel_threshold=False, input_scaling=True,
                                    activation_channel_equalization=True)

        return MixedPrecisionQuantizationConfig(qc)

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
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 256)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory == quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained KPI, "
            "final weights memory and total memory should be equal.")
        self.unit_test.assertTrue(quantization_info.final_kpi.activation_memory == 0,
                                  "Running weights only mixed-precision, final activation memory should be 0.")


class MixedPercisionSearchKPI4BitsAvgTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 4 bits on average
        return KPI(2544140 * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [1, 1]).all()
        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 16)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 16)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory == quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained activation and total KPI, "
            "final weights memory and total memory should be equal.")
        self.unit_test.assertTrue(quantization_info.final_kpi.activation_memory == 0,
                                  "Running weights only mixed-precision, final activation memory should be 0.")


class MixedPercisionSearchKPI2BitsAvgTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(2544200 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [2, 2]).all()
        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 4)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 4)

        # Verify final KPI
        self.unit_test.assertTrue(
            quantization_info.final_kpi.weights_memory == quantization_info.final_kpi.total_memory,
            "Running weights mixed-precision with unconstrained activation and total KPI, "
            "final weights memory and total memory should be equal.")
        self.unit_test.assertTrue(quantization_info.final_kpi.activation_memory == 0,
                                  "Running weights only mixed-precision, final activation memory should be 0.")


class MixedPercisionDepthwiseTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(30)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(weights_n_bits=16,
                                                 activation_n_bits=16)
        mp_tp_model = generate_mixed_precision_test_tp_model(base_cfg=base_config,
                                                             mp_bitwidth_candidates_list=[(8, 16), (2, 16), (4, 16),
                                                                                          (16, 16)])
        return generate_keras_tpc(name="mp_dw_test", tp_model=mp_tp_model)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=False,
                                    weights_bias_correction=False,
                                    weights_per_channel_threshold=True,
                                    input_scaling=False,
                                    activation_channel_equalization=False)

        return MixedPrecisionQuantizationConfig(qc)


class MixedPrecisionActivationDisabled(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=True,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    input_scaling=False,
                                    activation_channel_equalization=False)

        return MixedPrecisionQuantizationConfig(qc, num_of_images=1)

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        activation_disabled_config = base_config.clone_and_edit(enable_activation_quantization=False)

        mp_tp_model = generate_mixed_precision_test_tp_model(base_cfg=activation_disabled_config,
                                                             mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8)])
        return generate_keras_tpc(name="mp_weights_only_test", tp_model=mp_tp_model)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        assert (quantization_info.mixed_precision_cfg == [0,
                                                          0]).all()  # kpi is infinity -> should give best model - 8bits
        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[1].weights[0][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 256)