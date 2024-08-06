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

import torch
import numpy as np

from model_compression_toolkit.constants import LUT_VALUES, THRESHOLD, SIGNED, \
    LUT_VALUES_BITWIDTH
from model_compression_toolkit.core.pytorch.quantizer.lut_fake_quant import activation_lut_kmean_quantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest


class TestLUTQuantizerFakeQuantSigned(BasePytorchTest):

    def run_test(self, seed=0, **kwargs):
        threshold = 16
        lut_values = to_torch_tensor(np.array([-8.0, 0.0, 4.0]))
        tensor = to_torch_tensor(np.linspace(-1 * threshold, threshold, num=2 * threshold + 1))
        quantization_params = {SIGNED: True,
                               LUT_VALUES: lut_values,
                               THRESHOLD: threshold}

        # We divide the centers in 2^(8-1) (the minus 1 because of the signed quantization)
        div_val_output = (2 ** (LUT_VALUES_BITWIDTH - 1))

        # Construct the FakeQuant
        model = activation_lut_kmean_quantizer(activation_n_bits=8,  # dummy, not used in this function
                                               quantization_params=quantization_params)
        output = model(tensor)

        expected_unique_values = (lut_values / div_val_output) * threshold

        # Check expected unique values of the output
        self.unit_test.assertTrue((torch.unique(output) == expected_unique_values).all())

        # We expected each negative value in the input to be in the first center, each zero to be in the second
        # center and each positive value to be in the last center
        input_sign = torch.sign(tensor)
        self.unit_test.assertTrue((output[input_sign == -1] == expected_unique_values[0]).all())
        self.unit_test.assertTrue((output[input_sign == 0] == expected_unique_values[1]).all())
        self.unit_test.assertTrue((output[input_sign == 1] == expected_unique_values[2]).all())

class TestLUTQuantizerFakeQuantUnsigned(BasePytorchTest):

    def run_test(self, seed=0, **kwargs):
        threshold = 8
        lut_values = to_torch_tensor(np.array([0.0, 256.0]))
        tensor = to_torch_tensor(np.linspace(0, threshold, num=threshold+1))
        quantization_params = {SIGNED: False,
                               LUT_VALUES: lut_values,
                               THRESHOLD: threshold}

        # We divide the centers in 2^8
        div_val_output = (2 ** LUT_VALUES_BITWIDTH)

        # Construct the FakeQuant
        model = activation_lut_kmean_quantizer(activation_n_bits=8,  # dummy, not used in this function
                                               quantization_params=quantization_params)
        output = model(tensor)

        expected_unique_values = (lut_values / div_val_output) * threshold

        # Check expected unique values of the output
        self.unit_test.assertTrue((torch.unique(output) == expected_unique_values).all())

        # We expected each value that is lower or equal to 4 in the input to be in the first center and each value
        # bigger than that to be in the last center
        self.unit_test.assertTrue((output[tensor <= 4] == expected_unique_values[0]).all())
        self.unit_test.assertTrue((output[tensor > 4] == expected_unique_values[1]).all())


if __name__ == '__main__':
    unittest.main()
