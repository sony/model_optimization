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

from tests.quantizers_infrastructure_tests.pytorch_tests.base_pytorch_infrastructure_test import \
    BasePytorchInfrastructureTest, ZeroWeightsQuantizer, weight_quantization_config_uniform, ZeroActivationsQuantizer, \
    activations_quantization_config_uniform, weight_quantization_config, activations_quantization_config


class TestPytorchBaseQuantizer(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(weight_quantization_config_uniform)
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(activations_quantization_config_uniform)
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(activations_quantization_config_uniform)
        self.unit_test.assertEqual(f'Expect weight quantization got activation', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(weight_quantization_config_uniform)
        self.unit_test.assertEqual(f'Expect activation quantization got weight', str(e.exception))

        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == weight_quantization_config)

        quantizer = ZeroActivationsQuantizer(activations_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == activations_quantization_config)