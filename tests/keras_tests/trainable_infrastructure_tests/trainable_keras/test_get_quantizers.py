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

from model_compression_toolkit.trainable_infrastructure.common.get_quantizers import \
    get_trainable_quantizer_class
from tests.keras_tests.trainable_infrastructure_tests.base_keras_trainable_infra_test import \
    BaseKerasTrainableInfrastructureTest


class BaseKerasGetQuantizerTest(BaseKerasTrainableInfrastructureTest):

    def __init__(self, unit_test, quant_target, quant_method, quantizer_base_class, expected_quantizer_class=None):
        super().__init__(unit_test)

        self.quant_target = quant_target
        self.quant_method = quant_method
        self.quantizer_base_class = quantizer_base_class
        self.expected_quantizer_class = expected_quantizer_class


class TestGetTrainableQuantizer:

    def __init__(self, unit_test, quant_target, quant_method, quantizer_base_class, quantizer_id,
                 expected_quantizer_class=None):
        self.unit_test = unit_test
        self.quant_target = quant_target
        self.quant_method = quant_method
        self.quantizer_base_class = quantizer_base_class
        self.expected_quantizer_class = expected_quantizer_class
        self.quantizer_id = quantizer_id

    def run_test(self):
        quantizer = get_trainable_quantizer_class(quant_target=self.quant_target,
                                                  quantizer_id=self.quantizer_id,
                                                  quant_method=self.quant_method,
                                                  quantizer_base_class=self.quantizer_base_class)

        self.unit_test.assertTrue(issubclass(quantizer, self.quantizer_base_class))
        self.unit_test.assertEqual(quantizer, self.expected_quantizer_class)
