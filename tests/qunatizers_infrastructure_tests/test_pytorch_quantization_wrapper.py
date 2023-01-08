# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np
import unittest

import torch
import torch.nn as nn

from model_compression_toolkit import qunatizers_infrastructure as qi
from test_pytorch_base_quantizer import ZeroWeightsQuantizer, weight_quantization_config, ZeroActivationsQuantizer, activations_quantization_config


class TestPytorchQuantizationWrapper(unittest.TestCase):

    def test_pytorch_quantization_wrapper(self):
        # weights quantization
        nqd = qi.PytorchNodeQuantizationDispatcher()
        nqd.add_weight_quantizer('weight', ZeroWeightsQuantizer(weight_quantization_config))
        wrapper = qi.PytorchQuantizationWrapper(nn.Conv2d(3,20,3), nqd)
        (name, weight, quantizer) = wrapper._weight_vars[0]
        self.assertTrue(isinstance(wrapper, qi.PytorchQuantizationWrapper))
        self.assertTrue(isinstance(wrapper.layer, nn.Conv2d))
        self.assertTrue(name == 'weight')
        self.assertTrue((weight == getattr(wrapper.layer, 'weight')).any())
        self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(torch.Tensor(np.random.random((4, 3, 224, 224)))) # apply the wrapper on some random inputs
        self.assertTrue((0 == getattr(wrapper.layer, 'weight')).any()) # check the weight are now quantized
        self.assertTrue((y[0,:,0,0] == getattr(wrapper.layer, 'bias')).any()) # check the wrapper's outputs are equal to biases

        # activations quantization
        nqd = qi.PytorchNodeQuantizationDispatcher(activation_quantizers=[ZeroActivationsQuantizer(activations_quantization_config)])
        wrapper = qi.PytorchQuantizationWrapper(nn.Conv2d(3,20,3), nqd)
        (quantizer) = wrapper._activation_vars[0]
        self.assertTrue(isinstance(quantizer, ZeroActivationsQuantizer))
        y = wrapper(torch.Tensor(np.random.random((4, 3, 224, 224)))) # apply the wrapper on some random inputs
        self.assertTrue((y == 0).any()) # check the wrapper's outputs are equal to biases
