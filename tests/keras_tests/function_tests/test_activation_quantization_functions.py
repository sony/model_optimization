# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np

from model_compression_toolkit.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX

from model_compression_toolkit.core.keras.quantizer.fake_quant_builder import symmetric_quantization, \
    uniform_quantization


class TestActivationQuantizationFunctions(unittest.TestCase):

    def test_symmetric_quantization(self):
        for signed in [True, False]:
            activation_n_bits = 2
            quantization_params = {SIGNED: signed, THRESHOLD: 2}

            float_tensor = np.asarray([-2.1, -1.1, 0, 1, 1.1])
            if signed:
                expected_quantized_tensor = np.asarray([-2, -1, 0, 1, 1])
            else:
                expected_quantized_tensor = np.asarray([0, 0, 0, 1, 1])

            symmetric_quantization_fn = symmetric_quantization(activation_n_bits, quantization_params)
            quantized_tensor = symmetric_quantization_fn(float_tensor)
            self.assertTrue(np.array_equal(quantized_tensor, expected_quantized_tensor),
                            "Symmetric quantization failed.")

    def test_uniform_quantization(self):
        activation_n_bits = 2
        float_tensor = np.asarray([-2.1, -1.1, 0.0, 1.0, 1.1])
        params_dict = {'min_negative_max_positive': {RANGE_MIN: -2,
                                                     RANGE_MAX: 1,
                                                     'expected_quantized_tensor': np.asarray([-2, -1, 0, 1, 1])},
                       'min_positive': {RANGE_MIN: 1,
                                        RANGE_MAX: 4, #TODO: check why this is different than pytorch?
                                        'expected_quantized_tensor': np.asarray([0, 0, 0, 1, 1])},
                       'max_negative': {RANGE_MIN: -4, #TODO: check why this is different than pytorch?
                                        RANGE_MAX: -1,
                                        'expected_quantized_tensor': np.asarray([-2, -1, 0, 0, 0])}
                       }

        for test_name, test_args in params_dict.items():
            uniform_quantization_fn = uniform_quantization(activation_n_bits, {RANGE_MIN: test_args[RANGE_MIN],
                                                                               RANGE_MAX: test_args[RANGE_MAX]})
            quantized_tensor = uniform_quantization_fn(float_tensor)
            self.assertTrue(np.array_equal(quantized_tensor,
                                           test_args['expected_quantized_tensor']),
                            f"{test_name} - Uniform quantization test failed.")
