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
from keras.layers import Input, Conv2D, Activation
from keras.models import Model

from model_compression_toolkit.xquant.keras.model_analyzer import KerasModelAnalyzer
import model_compression_toolkit as mct
from tests.keras_tests.xquant_tests.test_xquant_end2end import random_data_gen


class TestKerasModelAnalyzer(unittest.TestCase):

    def setUp(self):
        self.float_model = self.get_model_to_test()
        self.repr_dataset = partial(random_data_gen, shape=self.get_input_shape())
        self.quantized_model, _ = mct.ptq.keras_post_training_quantization(in_model=self.float_model,
                                                                           representative_data_gen=self.repr_dataset)

        self.analyzer = KerasModelAnalyzer()
        self.float_name2quant_name = {
            self.float_model.layers[1].name: self.quantized_model.layers[2].name
        }

    def get_input_shape(self):
        return (8, 8, 3)

    def get_model_to_test(self):
        inputs = Input(shape=self.get_input_shape())
        x = Conv2D(3, 3, name='conv2d')(inputs)
        outputs = Activation('relu', name='activation')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def test_extract_model_activations(self):
        float_activations, quant_activations = self.analyzer.extract_model_activations(
            self.float_model, self.quantized_model, self.float_name2quant_name, next(self.repr_dataset())
        )
        self.assertIsInstance(float_activations, dict)
        self.assertIsInstance(quant_activations, dict)
        self.assertEqual(len(float_activations), 2) # conv + output
        self.assertEqual(len(quant_activations), 2)  # conv + output
        self.assertIn(self.float_model.layers[1].name, float_activations)
        self.assertIn(self.quantized_model.layers[2].name, quant_activations)

    def test_identify_quantized_compare_points(self):
        compare_points = self.analyzer.identify_quantized_compare_points(self.quantized_model)
        self.assertIsInstance(compare_points, list)
        self.assertEqual(len(compare_points), 1)
        self.assertIn(self.quantized_model.layers[2].name, compare_points)

    def test_find_corresponding_float_layer(self):
        corresponding_layer = self.analyzer.find_corresponding_float_layer(self.quantized_model.layers[2].name, self.quantized_model)
        self.assertEqual(corresponding_layer, self.float_model.layers[1].name)

    def test_extract_float_layer_names(self):
        layer_names = self.analyzer.extract_float_layer_names(self.float_model)
        self.assertIsInstance(layer_names, list)
        self.assertIn('conv2d', layer_names)
        self.assertIn('activation', layer_names)

if __name__ == '__main__':
    unittest.main()

