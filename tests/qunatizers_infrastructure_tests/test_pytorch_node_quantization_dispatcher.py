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
import unittest


from model_compression_toolkit import qunatizers_infrastructure as qi
from test_pytorch_base_quantizer import ZeroWeightsQuantizer, weight_quantization_config, ZeroActivationsQuantizer, activations_quantization_config


class TestPytorchNodeQuantizationDispatcher(unittest.TestCase):

    def test_pytorch_node_quantization_dispatcher(self):
        nqd = qi.PytorchNodeQuantizationDispatcher()
        self.assertFalse(nqd.is_weights_quantization)
        nqd.add_weight_quantizer('weight', ZeroWeightsQuantizer(weight_quantization_config))
        self.assertTrue(nqd.is_weights_quantization)
        self.assertFalse(nqd.is_activation_quantization)
        self.assertTrue(isinstance(nqd.weight_quantizers.get('weight'), ZeroWeightsQuantizer))

        nqd = qi.PytorchNodeQuantizationDispatcher(activation_quantizers=[ZeroActivationsQuantizer(activations_quantization_config)])
        self.assertFalse(nqd.is_weights_quantization)
        self.assertTrue(nqd.is_activation_quantization)
        self.assertTrue(isinstance(nqd.activation_quantizers[0], ZeroActivationsQuantizer))
