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
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_keras_tpc
import model_compression_toolkit as mct
import tensorflow as tf

from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np

keras = tf.keras
layers = keras.layers


class ReusedLayerMixedPrecisionTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(weights_n_bits=16,
                                                 activation_n_bits=16)
        mp_tp_model = generate_mixed_precision_test_tp_model(base_cfg=base_config,
                                                             mp_bitwidth_candidates_list=[(2, 16), (4, 16), (16, 16)])
        return generate_keras_tpc(name="reused_layer_mp_test", tp_model=mp_tp_model)



    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=True,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    input_scaling=True,
                                    activation_channel_equalization=True)

        return MixedPrecisionQuantizationConfig(qc)

    def get_mixed_precision_v2_config(self):
        return MixedPrecisionQuantizationConfigV2()

    def create_networks(self):
        layer = layers.Conv2D(3, 4)
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layer(inputs)
        x = layer(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def get_kpi(self):
        return KPI(np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        if isinstance(float_model.layers[1], layers.Conv2D):
            self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.Conv2D))
            self.unit_test.assertFalse(hasattr(quantized_model.layers[2], 'input_shape'))  # assert it's reused
        if isinstance(float_model.layers[1], layers.SeparableConv2D):
            self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.DepthwiseConv2D))
            self.unit_test.assertFalse(hasattr(quantized_model.layers[2], 'input_shape'))  # assert it's reused
            self.unit_test.assertTrue(isinstance(quantized_model.layers[4], layers.Conv2D))
            self.unit_test.assertFalse(hasattr(quantized_model.layers[4], 'input_shape'))  # assert it's reused


class ReusedSeparableMixedPrecisionTest(ReusedLayerMixedPrecisionTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        layer = layers.SeparableConv2D(3, 3)
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layer(inputs)
        x = layer(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model
