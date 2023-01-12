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
    BasePytorchInfrastructureTest, ZeroWeightsQuantizer, ZeroActivationsQuantizer


class TestPytorchNodeWeightsQuantizationDispatcher(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        nqd = self.get_dispatcher()
        self.unit_test.assertFalse(nqd.is_weights_quantization)
        nqd.add_weight_quantizer('weight', ZeroWeightsQuantizer(self.get_weights_quantization_config()))
        self.unit_test.assertTrue(nqd.is_weights_quantization)
        self.unit_test.assertFalse(nqd.is_activation_quantization)
        self.unit_test.assertTrue(isinstance(nqd.weight_quantizers.get('weight'), ZeroWeightsQuantizer))


class TestPytorchNodeActivationQuantizationDispatcher(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        nqd = self.get_dispatcher(activation_quantizers=[ZeroActivationsQuantizer(self.get_activation_quantization_config())])
        self.unit_test.assertFalse(nqd.is_weights_quantization)
        self.unit_test.assertTrue(nqd.is_activation_quantization)
        self.unit_test.assertTrue(isinstance(nqd.activation_quantizers[0], ZeroActivationsQuantizer))
