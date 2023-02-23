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
import numpy as np
import torch

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizer_utils import \
    get_working_device
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers import \
    ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer, ActivationUniformInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest


class TestPytorchActivationsSymmetricInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        thresholds = np.asarray([4])
        num_bits = 2
        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=True)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < thresholds[
                   0], f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= -thresholds[
            0], f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.unit_test.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = torch.Tensor(thresholds).to(get_working_device())
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


class TestPytorchActivationsUnsignedSymmetricInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        thresholds = np.asarray([4])
        num_bits = 2
        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=False)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < thresholds[
                   0], f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.unit_test.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert all values are non-negative (unsigned quantization)
        self.unit_test.assertTrue(torch.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = thresholds[0]
        scale = thresholds / (2 ** num_bits)
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), 0, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


class TestPytorchActivationsPOTInferableQuantizerRaise(BaseInferableQuantizerTest):
    
    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationPOTInferableQuantizer(num_bits=8,
                                            # Not POT threshold
                                            threshold=np.asarray([3]),
                                            signed=True)
        self.unit_test.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ActivationPOTInferableQuantizer(num_bits=8,
                                            # Not float
                                            threshold=4,
                                            signed=True)

        self.unit_test.assertEqual('Threshold is expected to be numpy array, but is of type <class \'int\'>', str(e.exception))


class TestPytorchActivationsSignedPOTInferableQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        thresholds = np.asarray([1])
        num_bits = 2
        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    signed=True,
                                                    threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device())).dequantize()

        assert torch.max(
            fake_quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(
            fake_quantized_tensor) >= -thresholds[0], f'Quantized values should not contain values lower than threshold'
        # Expect to have no more than 2**num_bits unique values
        self.unit_test.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = thresholds[0]
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))


class TestPytorchActivationsUnsignedPOTInferableQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        thresholds = np.asarray([1])
        num_bits = 2
        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    signed=False,
                                                    threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device())).dequantize()

        assert torch.max(
            fake_quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than threshold'
        # Expect to have no more than 2**num_bits unique values
        self.unit_test.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')

        # Assert manually quantized values are the same:
        thresholds = thresholds[0]
        scale = thresholds / (2 ** num_bits)
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), 0, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))


class TestPytorchActivationsUniformInferableQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        min_range = np.asarray([-10])
        max_range = np.asarray([5])
        num_bits = 2
        quantizer = ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                        min_range=min_range,
                                                        max_range=max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) <= max_range[
                   0], f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= min_range[
                   0], f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.unit_test.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        max_range = max_range[0]
        min_range = min_range[0]
        scale = (max_range - min_range) / (2 ** num_bits - 1)

        manually_quantized_tensor = torch.round((torch.clip(input_tensor.to(get_working_device()), min_range,
                                                            max_range) - min_range) / scale) * scale + min_range
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


class TestPytorchActivationsUniformInferableZeroNotInRange(BaseInferableQuantizerTest):
    
    def run_test(self):
        min_range = np.asarray([3])
        max_range = np.asarray([10])
        num_bits = 2
        quantizer = ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                        min_range=min_range,
                                                        max_range=max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) <= max_range[
                   0], f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.unit_test.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')

        # Assert that quantization range was fixed to include 0
        self.unit_test.assertTrue(0 in quantized_tensor.unique(),
                        f'Expected to find 0 in quantized values but unique values are {quantized_tensor.unique()}')

        # Assert manually quantized values are the same:
        max_range = max_range[0] if max_range[0] >= 0 else 0
        min_range = min_range[0] if min_range[0] <= 0 else 0
        scale = (max_range - min_range) / (2 ** num_bits - 1)

        manually_quantized_tensor = torch.round((torch.clip(input_tensor.to(get_working_device()), min_range,
                                                            max_range) - min_range) / scale) * scale + min_range
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))
