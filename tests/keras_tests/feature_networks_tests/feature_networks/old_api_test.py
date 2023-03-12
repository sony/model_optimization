# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import tensorflow as tf
import numpy as np

import model_compression_toolkit as mct
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs


keras = tf.keras
layers = keras.layers


class OldApiTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, mp_enable=False, gptq_enable=False):
        super().__init__(unit_test, val_batch_size=1, num_calibration_iter=100)
        self.mp_enable = mp_enable
        self.gptq_enable = gptq_enable

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(weights_n_bits=16,
                                                 activation_n_bits=16)

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             mp_bitwidth_candidates_list=[(8, 16), (2, 16), (4, 16), (16, 16)],
                                             name="old_api_test")

    def get_kpi(self):
        return mct.KPI()

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = layers.Conv2D(1, 1)(inputs)
        outputs = layers.ReLU()(outputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def run_test(self, experimental_exporter=False):
        model_float = self.create_networks()
        core_config = self.get_core_config()
        quant_config = core_config.quantization_config
        gptq_config = mct.gptq.GradientPTQConfig(1, keras.optimizers.Adam(learning_rate=1e-12)) if self.gptq_enable else None
        if self.mp_enable:
            quant_config = mct.MixedPrecisionQuantizationConfig(quant_config, num_of_images=1)
            facade_fn = mct.keras_post_training_quantization_mixed_precision
            ptq_model, quantization_info = facade_fn(model_float,
                                                     self.representative_data_gen,
                                                     self.get_kpi(),
                                                     n_iter=self.num_calibration_iter,
                                                     quant_config=quant_config,
                                                     gptq_config=gptq_config,
                                                     target_platform_capabilities=self.get_tpc(),
                                                     )
        else:
            facade_fn = mct.keras_post_training_quantization
            ptq_model, quantization_info = facade_fn(model_float,
                                                     self.representative_data_gen,
                                                     n_iter=self.num_calibration_iter,
                                                     quant_config=quant_config,
                                                     gptq_config=gptq_config,
                                                     target_platform_capabilities=self.get_tpc(),
                                                     )

        self.compare(ptq_model, model_float, input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quant_model, float_model, input_x=None, quantization_info=None):
        out_float = float_model(input_x[0]).numpy()
        out_quant = quant_model(input_x[0]).numpy()
        self.unit_test.assertTrue(np.isclose(np.linalg.norm(np.abs(out_float-out_quant)), 0, atol=0.01))
