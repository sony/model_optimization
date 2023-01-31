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

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from model_compression_toolkit.quantizers_infrastructure.keras.config_serialization import config_serialization, \
    config_deserialization
from tests.quantizers_infrastructure_tests.keras_tests.base_keras_infrastructure_test import \
    BaseKerasInfrastructureTest, ZeroWeightsQuantizer, ZeroActivationsQuantizer


class TestKerasBaseWeightsQuantizer(BaseKerasInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_weights_quantization_config(self):
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.UNIFORM,
                                               weights_n_bits=8,
                                               weights_quantization_params={},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=3,
                                               weights_per_channel_threshold=True,
                                               min_threshold=0)

    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, '
                                   f''f'<QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM',
                                   str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Expect weight quantization got activation', str(e.exception))

        weight_quantization_config = super(TestKerasBaseWeightsQuantizer, self).get_weights_quantization_config()
        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == weight_quantization_config)
        config_data = config_serialization(weight_quantization_config)
        self.unit_test.assertTrue(config_data['enable_weights_quantization'])
        deserialized_config = config_deserialization(config_data)
        self.unit_test.assertTrue(weight_quantization_config.__dict__ == deserialized_config.__dict__)

class TestKerasBaseActivationsQuantizer(BaseKerasInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_activation_quantization_config(self):
        return TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod.UNIFORM,
                                                  activation_n_bits=8,
                                                  activation_quantization_params={},
                                                  enable_activation_quantization=True,
                                                  min_threshold=0)

    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, '
                                   f'<QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM',
                                   str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Expect activation quantization got weight', str(e.exception))

        activation_quantization_config = super(TestKerasBaseActivationsQuantizer,
                                               self).get_activation_quantization_config()
        quantizer = ZeroActivationsQuantizer(activation_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == activation_quantization_config)
        config_data = config_serialization(activation_quantization_config)
        self.unit_test.assertTrue(config_data['enable_activation_quantization'])
        deserialized_config = config_deserialization(config_data)
        self.unit_test.assertTrue(activation_quantization_config.__dict__ == deserialized_config.__dict__)
