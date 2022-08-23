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

from model_compression_toolkit import KPI, MixedPrecisionQuantizationConfigV2
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input, Add

from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_methods import weights_size_kpi
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
import tensorflow as tf
import numpy as np

from tests.keras_tests.tpc_keras import generate_activation_mp_tpc_keras

keras = tf.keras
layers = keras.layers


def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class BaseMixedPrecisionBopsTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, mixed_precision_candidates_list):
        super().__init__(unit_test)

        self.mixed_precision_candidates_list = mixed_precision_candidates_list

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        mp_tp_model = generate_tp_model_with_activation_mp(base_config, self.mixed_precision_candidates_list)

        return generate_activation_mp_tpc_keras(tp_model=mp_tp_model)

    def get_mixed_precision_v2_config(self):
        return MixedPrecisionQuantizationConfigV2(num_of_images=1)

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, 3]]

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that some layers got bit-width smaller than 8 bits (so checking candidate index is not 0)
        self.unit_test.assertTrue(all(i > 0 for i in quantization_info.mixed_precision_cfg))
        # Verify final BOPs KPI
        self.unit_test.assertTrue(quantization_info.final_kpi.bops <= self.get_kpi().bops)


class MixedPrecisionBopsBasicTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test):

        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_networks(self):
        inputs = Input(shape=self.get_input_shapes()[0][1:])
        x = Conv2D(3, 4)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        outputs = Conv2D(3, 4)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_kpi(self):
        return KPI(bops=1000000)  # should require some quantization to all layers


class MixedPrecisionBopsAllWeightsLayersTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test, mixed_precision_candidates_list=None):

        if mixed_precision_candidates_list is None:
            mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_networks(self):
        inputs = Input(shape=self.get_input_shapes()[0][1:])
        x = Conv2D(3, 4)(inputs)
        x = Conv2D(3, 4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(3, 4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = DepthwiseConv2D(3, depth_multiplier=5)(x)
        outputs = Dense(5)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_kpi(self):
        return KPI(bops=1252512)  # should require some quantization to all layers


class MixedPrecisionWeightsOnlyBopsTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, mixed_precision_candidates_list=[(8, 8), (4, 8), (2, 8)])

    def get_kpi(self):
        return KPI(bops=5010100)  # should require some quantization to all layers


class MixedPrecisionActivationOnlyBopsTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, mixed_precision_candidates_list=[(8, 8), (8, 4), (8, 2)])

    def get_kpi(self):
        return KPI(bops=5010100)  # should require some quantization to all layers

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that some layers got bit-width smaller than 8 bits (so checking candidate index is not 0)
        self.unit_test.assertTrue(any(i > 0 for i in quantization_info.mixed_precision_cfg))
        # Verify final BOPs KPI
        self.unit_test.assertTrue(quantization_info.final_kpi.bops <= self.get_kpi().bops)


class MixedPrecisionBopsAndWeightsKPITest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(weights_memory=170, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsAndActivationKPITest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(activation_memory=460, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsAndTotalKPITest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(total_memory=650, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsWeightsActivationKPITest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(weights_memory=200, activation_memory=500, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsMultipleOutEdgesTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test):

        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_networks(self):
        inputs = Input(shape=self.get_input_shapes()[0][1:])
        x = Conv2D(3, 4)(inputs)
        y = Conv2D(3, 4)(inputs)
        x_relu = ReLU()(x)
        y_relu = ReLU()(y)
        outputs = Add()([x_relu, y_relu])
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_kpi(self):
        return KPI(bops=1)  # No layers with BOPs count

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that all layers got 8 bits (so checking candidate index is 0)
        self.unit_test.assertTrue(all(i == 0 for i in quantization_info.mixed_precision_cfg))
