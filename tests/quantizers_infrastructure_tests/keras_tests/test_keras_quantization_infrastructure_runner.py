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
import unittest
import tensorflow as tf

from tests.quantizers_infrastructure_tests.keras_tests.test_keras_base_quantizer import TestKerasBaseWeightsQuantizer, \
    TestKerasBaseActivationsQuantizer
from tests.quantizers_infrastructure_tests.keras_tests.test_keras_node_quantization_dispatcher import \
    TestKerasNodeWeightsQuantizationDispatcher, TestKerasNodeActivationsQuantizationDispatcher
from tests.quantizers_infrastructure_tests.keras_tests.test_keras_quantization_wrapper import \
    TestKerasWeightsQuantizationWrapper, TestKerasActivationsQuantizationWrapper

layers = tf.keras.layers


class KerasInfrastructureTest(unittest.TestCase):

    def test_layer_keras_infrastructure(self):
        TestKerasWeightsQuantizationWrapper(self).run_test()
        TestKerasActivationsQuantizationWrapper(self).run_test()

    def test_keras_node_quantization_dispatcher(self):
        TestKerasNodeWeightsQuantizationDispatcher(self).run_test()
        TestKerasNodeActivationsQuantizationDispatcher(self).run_test()

    def test_keras_base_quantizer(self):
        TestKerasBaseWeightsQuantizer(self).run_test()
        TestKerasBaseActivationsQuantizer(self).run_test()


if __name__ == '__main__':
    unittest.main()
