# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.user_info import UserInformation
from tests.keras_tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MixedPercisionBaseTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                    mct.ThresholdSelectionMethod.MSE,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=True,
                                    relu_unbound_correction=True,
                                    input_scaling=True)

        return MixedPrecisionQuantizationConfig(qc, weights_n_bits=[2, 8, 4], num_of_images=100)

    def get_bit_widths_config(self):
        return None

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
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

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                    mct.ThresholdSelectionMethod.MSE,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=False,
                                    activation_channel_equalization=True,
                                    relu_unbound_correction=True,
                                    input_scaling=True)

        return MixedPrecisionQuantizationConfig(qc, weights_n_bits=[8, 2, 3])

    def get_bit_widths_config(self):
        # First layer should be quantized using 2 bits
        # Second layer should be quantized using 3 bits
        return [2, 1]

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


class MixedPercisionDepthwiseTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
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

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.ThresholdSelectionMethod.MSE,
                                    mct.ThresholdSelectionMethod.MSE,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    mct.QuantizationMethod.POWER_OF_TWO,
                                    weights_bias_correction=False,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=False,
                                    relu_unbound_correction=False,
                                    input_scaling=False,
                                    activation_n_bits=16)

        return MixedPrecisionQuantizationConfig(qc, weights_n_bits=[2, 8, 4, 16])
