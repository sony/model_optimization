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

import numpy as np
import torch

from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import get_working_device


class TestActivationSymmetricQuantizer(unittest.TestCase):

    def test_activation_signed_symmetric_inferable_quantizer(self):
        quantizer = qi.pytorch_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=2,
                                                                                          threshold=np.asarray([4]),
                                                                                          signed=True)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < 4, f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= -4, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_activation_unsigned_symmetric_inferable_quantizer(self):
        quantizer = qi.pytorch_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=2,
                                                                                          threshold=np.asarray([4]),
                                                                                          signed=False)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < 4, f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(torch.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')


class TestActivationPOTQuantizer(unittest.TestCase):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=8,
                                                                            # Not POT threshold
                                                                            threshold=np.asarray([3]),
                                                                            signed=True)
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=8,
                                                                            # Not float
                                                                            threshold=4,
                                                                            signed=True)

        self.assertEqual('Threshold is expected to be numpy array, but is of type <class \'int\'>', str(e.exception))

    def test_pot_signed_inferable_quantizer(self):
        quantizer = qi.pytorch_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=2,
                                                                                    signed=True,
                                                                                    threshold=np.asarray([1]))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device())).dequantize()

        assert torch.max(
            fake_quantized_tensor) < 1, f'Quantized values should not contain values greater than threshold'
        assert torch.min(
            fake_quantized_tensor) >= -1, f'Quantized values should not contain values lower than threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

    def test_pot_unsigned_inferable_quantizer(self):
        quantizer = qi.pytorch_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=2,
                                                                                    signed=False,
                                                                                    threshold=np.asarray([1]))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device())).dequantize()

        assert torch.max(
            fake_quantized_tensor) < 1, f'Quantized values should not contain values greater than threshold'
        assert torch.min(fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')


class TestActivationUniformQuantizer(unittest.TestCase):

    def test_uniform_inferable_quantizer(self):
        quantizer = qi.pytorch_inferable_quantizers.ActivationUniformInferableQuantizer(num_bits=2,
                                                                                        min_range=np.asarray([-10]),
                                                                                        max_range=np.asarray([5]))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) <= 5, f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= -10, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_uniform_inferable_quantizer_zero_not_in_range(self):
        quantizer = qi.pytorch_inferable_quantizers.ActivationUniformInferableQuantizer(num_bits=2,
                                                                                        min_range=np.asarray([3]),
                                                                                        max_range=np.asarray([10]))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) <= 10, f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')

        # Assert that quantization range was fixed to include 0
        self.assertTrue(0 in quantized_tensor.unique(),
                        f'Expected to find 0 in quantized values but unique values are {quantized_tensor.unique()}')
