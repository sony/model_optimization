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


import unittest

from tests.quantizers_infrastructure_tests.pytorch_tests.test_pytorch_base_quantizer import \
    TestPytorchBaseWeightsQuantizer, TestPytorchBaseActivationQuantizer
from tests.quantizers_infrastructure_tests.pytorch_tests.test_pytorch_node_quantization_dispatcher import \
    TestPytorchNodeActivationQuantizationDispatcher, TestPytorchNodeWeightsQuantizationDispatcher
from tests.quantizers_infrastructure_tests.pytorch_tests.test_pytorch_quantization_wrapper import \
    TestPytorchWeightsQuantizationWrapper, TestPytorchActivationQuantizationWrapper


class PytorchInfrastructureTest(unittest.TestCase):

    def test_layer_pytorch_infrastructre(self):
        TestPytorchWeightsQuantizationWrapper(self).run_test()
        TestPytorchActivationQuantizationWrapper(self).run_test()

    def test_pytorch_node_quantization_dispatcher(self):
        TestPytorchNodeWeightsQuantizationDispatcher(self).run_test()
        TestPytorchNodeActivationQuantizationDispatcher(self).run_test()

    def test_pytorch_base_quantizer(self):
        TestPytorchBaseWeightsQuantizer(self).run_test()
        TestPytorchBaseActivationQuantizer(self).run_test()

if __name__ == '__main__':
    unittest.main()