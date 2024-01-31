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
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_keras_tpc
import model_compression_toolkit as mct
import tensorflow as tf

from tests.common_tests.helpers.generate_test_tp_model import generate_test_op_qc, generate_test_attr_configs
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np

from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class ReusedLayerMixedPrecisionTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, experimental_exporter=True)

    def get_tpc(self):
        base_config = generate_test_op_qc(activation_n_bits=16,
                                          **generate_test_attr_configs(default_cfg_nbits=16,
                                                                       kernel_cfg_nbits=16))

        default_config = base_config.clone_and_edit(attr_weights_configs_mapping={})

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             default_config=default_config,
                                             mp_bitwidth_candidates_list=[(2, 16), (4, 16), (16, 16)],
                                             name="reused_layer_mp_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=True, weights_bias_correction=True,
                                           input_scaling=True, activation_channel_equalization=True)

    def get_mixed_precision_v2_config(self):
        return MixedPrecisionQuantizationConfigV2()

    def create_networks(self):
        layer = layers.Conv2D(3, 4)
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layer(inputs)
        x = layer(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def get_kpi(self):
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        if isinstance(float_model.layers[1], layers.Conv2D):
            conv_layer = get_layers_from_model_by_type(quantized_model, layers.Conv2D)[0]
            self.unit_test.assertFalse(hasattr(conv_layer, 'input_shape'))  # assert it's reused
        if isinstance(float_model.layers[1], layers.SeparableConv2D):
            dw_layer = get_layers_from_model_by_type(quantized_model, layers.DepthwiseConv2D)[0]
            self.unit_test.assertFalse(hasattr(dw_layer, 'input_shape'))  # assert it's reused
            conv_layer = get_layers_from_model_by_type(quantized_model, layers.Conv2D)[0]
            self.unit_test.assertFalse(hasattr(conv_layer, 'input_shape'))  # assert it's reused


class ReusedSeparableMixedPrecisionTest(ReusedLayerMixedPrecisionTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        layer = layers.SeparableConv2D(3, 3)
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layer(inputs)
        x = layer(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model
