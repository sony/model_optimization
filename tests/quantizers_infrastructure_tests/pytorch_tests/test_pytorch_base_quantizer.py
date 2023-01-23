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
from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from tests.quantizers_infrastructure_tests.pytorch_tests.base_pytorch_infrastructure_test import \
    BasePytorchInfrastructureTest, ZeroWeightsQuantizer, ZeroActivationsQuantizer
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from model_compression_toolkit import QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class TestPytorchBaseWeightsQuantizer(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_weights_quantization_config(self):
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.UNIFORM,
                                               weights_n_bits=8,
                                               weights_quantization_params={},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=-1,
                                               weights_per_channel_threshold=True,
                                               min_threshold=0)

    def run_test(self):

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Expect weight quantization got activation', str(e.exception))

        weight_quantization_config = super(TestPytorchBaseWeightsQuantizer, self).get_weights_quantization_config()
        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == weight_quantization_config)


class TestPytorchBaseActivationQuantizer(BasePytorchInfrastructureTest):

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
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Expect activation quantization got weight', str(e.exception))

        activation_quantization_config = super(TestPytorchBaseActivationQuantizer, self).get_activation_quantization_config()
        quantizer = ZeroActivationsQuantizer(activation_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == activation_quantization_config)