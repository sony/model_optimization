#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from functools import partial

import unittest
from model_compression_toolkit.core.pytorch.utils import get_working_device, to_torch_tensor

from model_compression_toolkit.xquant.pytorch.model_analyzer import PytorchModelAnalyzer
from tests.pytorch_tests.xquant_tests.test_xquant_end2end import random_data_gen
import model_compression_toolkit as mct
from torch import nn
class TestPytorchModelAnalyzer(unittest.TestCase):

    def setUp(self):
        self.device = get_working_device()
        self.float_model = self.get_model_to_test()
        self.repr_dataset = partial(random_data_gen, shape=self.get_input_shape())
        self.quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=self.float_model,
                                                                             representative_data_gen=self.repr_dataset)

        self.float_model.to(self.device)
        self.quantized_model.to(self.device)

        self.analyzer = PytorchModelAnalyzer()
        self.float_name2quant_name = {
            'conv': 'conv'
        }

    def get_input_shape(self):
        return (3, 8, 8)

    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(3, 3, 3, 1)
                self.activation = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.activation(x)
                return x

        return Model()

    def test_extract_model_activations(self):
        float_activations, quant_activations = self.analyzer.extract_model_activations(
            self.float_model, self.quantized_model, self.float_name2quant_name, to_torch_tensor(next(self.repr_dataset()))
        )
        self.assertIsInstance(float_activations, dict)
        self.assertIsInstance(quant_activations, dict)
        self.assertEqual(len(float_activations), 2) # conv + output
        self.assertEqual(len(quant_activations), 2)  # conv + output
        self.assertIn('conv', float_activations)
        self.assertIn('conv', quant_activations)

    def test_identify_quantized_compare_points(self):
        compare_points = self.analyzer.identify_quantized_compare_points(self.quantized_model)
        self.assertIsInstance(compare_points, list)
        self.assertEqual(len(compare_points), 1)
        self.assertIn('conv', compare_points)

    def test_find_corresponding_float_layer(self):
        corresponding_layer = self.analyzer.find_corresponding_float_layer('conv', self.quantized_model)
        self.assertEqual(corresponding_layer, 'conv')

    def test_extract_float_layer_names(self):
        layer_names = self.analyzer.extract_float_layer_names(self.float_model)
        self.assertIsInstance(layer_names, list)
        self.assertIn('conv', layer_names)
        self.assertIn('activation', layer_names)

if __name__ == '__main__':
    unittest.main()
