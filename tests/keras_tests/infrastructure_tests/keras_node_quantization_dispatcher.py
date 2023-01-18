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

from tests.keras_tests.infrastructure_tests.base_keras_infrastructure_test import BaseKerasInfrastructureTest, \
    IdentityQuantizer, weight_quantization_config, ZeroActivationsQuantizer, \
    activation_quantization_config

keras = tf.keras
layers = keras.layers


class TestKerasNodeQuantizationDispatcher(BaseKerasInfrastructureTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):

        nqd = self.get_dispatcher()
        self.unit_test.assertFalse(nqd.is_weights_quantization)
        nqd.add_weight_quantizer('weight', IdentityQuantizer(weight_quantization_config))
        self.unit_test.assertTrue(nqd.is_weights_quantization)
        self.unit_test.assertFalse(nqd.is_activation_quantization)
        self.unit_test.assertTrue(isinstance(nqd.weight_quantizers.get('weight'), IdentityQuantizer))

        nqd = self.get_dispatcher(activation_quantizers=[ZeroActivationsQuantizer(activation_quantization_config)])
        self.unit_test.assertFalse(nqd.is_weights_quantization)
        self.unit_test.assertTrue(nqd.is_activation_quantization)
        self.unit_test.assertTrue(isinstance(nqd.activation_quantizers[0], ZeroActivationsQuantizer))
