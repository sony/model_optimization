# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import NoOpQuantizeConfig

from tests.keras_tests.tpc_keras import get_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers


class QuantizationAwareTrainingTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, weight_bits=2, activation_bits=4):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        super().__init__(unit_test)

    def get_tpc(self):
        return get_tpc("QAT_test", weight_bits=self.weight_bits, activation_bits=self.activation_bits)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = layers.Conv2D(3, 4, activation='relu')(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def run_test(self):
        model_float = self.create_networks()
        qc = self.get_quantization_config()
        ptq_model, quantization_info, custom_opjects = mct.keras_quantization_aware_training_init(model_float,
                                                                                                  self.representative_data_gen,
                                                                                                  fw_info=self.get_fw_info(),
                                                                                                  target_platform_capabilities=self.get_tpc())

        self.compare(ptq_model,
                     model_float,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2].layer, layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].layer, layers.Activation))
        _, qconfig = quantized_model.layers[2].quantize_config.get_weights_and_quantizers(quantized_model.layers[2].layer)[0]
        self.unit_test.assertTrue(qconfig.num_bits == self.weight_bits)
        self.unit_test.assertTrue(isinstance(quantized_model.layers[3].quantize_config, NoOpQuantizeConfig))
