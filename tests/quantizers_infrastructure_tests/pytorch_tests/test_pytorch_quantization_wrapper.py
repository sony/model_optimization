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

import torch
import torch.nn as nn

from model_compression_toolkit import quantizers_infrastructure as qi
from tests.quantizers_infrastructure_tests.pytorch_tests.base_pytorch_infrastructure_test import \
    ZeroWeightsQuantizer, ZeroActivationsQuantizer
from tests.quantizers_infrastructure_tests.pytorch_tests.base_pytorch_infrastructure_test import \
    BasePytorchInfrastructureTest


class TestPytorchWeightsQuantizationWrapper(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_layer(self):
        return nn.Conv2d(3,20,3)

    def run_test(self):
        nqd = self.get_dispatcher()
        nqd.add_weight_quantizer('weight', ZeroWeightsQuantizer(self.get_weights_quantization_config()))
        wrapper = self.get_wrapper(self.create_layer(), nqd)
        (name, weight, quantizer) = wrapper._weight_vars[0]
        self.unit_test.assertTrue(isinstance(wrapper, qi.PytorchQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(wrapper.layer, nn.Conv2d))
        self.unit_test.assertTrue(name == 'weight')
        self.unit_test.assertTrue((weight == getattr(wrapper.layer, 'weight')).any())
        self.unit_test.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(torch.Tensor(self.generate_inputs()[0])) # apply the wrapper on some random inputs
        self.unit_test.assertTrue((0 == getattr(wrapper.layer, 'weight')).any()) # check the weight are now quantized
        self.unit_test.assertTrue((y[0,:,0,0] == getattr(wrapper.layer, 'bias')).any()) # check the wrapper's outputs are equal to biases


class TestPytorchActivationQuantizationWrapper(TestPytorchWeightsQuantizationWrapper):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        nqd = self.get_dispatcher(activation_quantizers=[ZeroActivationsQuantizer(self.get_activation_quantization_config())])
        wrapper = qi.PytorchQuantizationWrapper(self.create_layer(), nqd)
        (quantizer) = wrapper._activation_vars[0]
        self.unit_test.assertTrue(isinstance(quantizer, ZeroActivationsQuantizer))
        # y = wrapper(torch.Tensor(np.random.random((4, 3, 224, 224)))) # apply the wrapper on some random inputs
        y = wrapper(torch.Tensor(self.generate_inputs()[0])) # apply the wrapper on some random inputs
        self.unit_test.assertTrue((y == 0).any()) # check the wrapper's outputs are equal to biases
