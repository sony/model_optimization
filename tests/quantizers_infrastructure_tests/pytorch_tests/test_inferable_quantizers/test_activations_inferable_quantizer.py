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
import warnings

import numpy as np
import torch

from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import get_working_device


class TestActivationSymmetricQuantizer(unittest.TestCase):

    def test_activation_signed_symmetric_inferable_quantizer(self):
        thresholds = np.asarray([4])
        num_bits = 2
        quantizer = qi.pytorch_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                                                          threshold=thresholds,
                                                                                          signed=True)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < thresholds[0], f'Quantized values should not contain ' \
                                               f'values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= -thresholds[0], f'Quantized values should not contain ' \
                                                 f'values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = torch.Tensor(thresholds).to(get_working_device())
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))

    def test_activation_unsigned_symmetric_inferable_quantizer(self):
        thresholds = np.asarray([4])
        num_bits = 2
        quantizer = qi.pytorch_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                                                          threshold=thresholds,
                                                                                          signed=False)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < thresholds[0], f'Quantized values should not contain ' \
                                               f'values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(torch.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = thresholds[0]
        scale = thresholds / (2 ** num_bits)
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), 0, thresholds - scale) / scale) * scale
        self.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


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
        thresholds = np.asarray([1])
        num_bits = 2
        quantizer = qi.pytorch_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                                                    signed=True,
                                                                                    threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        assert torch.max(
            fake_quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(
            fake_quantized_tensor) >= -thresholds[0], f'Quantized values should not contain values lower than threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = thresholds[0]
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))

    def test_pot_unsigned_inferable_quantizer(self):
        thresholds = np.asarray([1])
        num_bits = 2
        quantizer = qi.pytorch_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                                                    signed=False,
                                                                                    threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        assert torch.max(
            fake_quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')

        # Assert manually quantized values are the same:
        thresholds = thresholds[0]
        scale = thresholds / (2 ** num_bits)
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), 0, thresholds - scale) / scale) * scale
        self.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))


class TestActivationUniformQuantizer(unittest.TestCase):

    def test_uniform_inferable_quantizer(self):
        min_range = np.asarray([-10])
        max_range = np.asarray([5])
        num_bits = 2
        quantizer = qi.pytorch_inferable_quantizers.ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                                                        min_range=min_range,
                                                                                        max_range=max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) <= max_range[0], f'Quantized values should not contain values greater than maximal ' \
                                               f'threshold '
        assert torch.min(
            quantized_tensor) >= min_range[0], f'Quantized values should not contain values ' \
                                               f'lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        max_range = max_range[0]
        min_range = min_range[0]
        scale = (max_range - min_range) / (2 ** num_bits - 1)

        manually_quantized_tensor = torch.round((torch.clip(input_tensor.to(get_working_device()), min_range,
                                                            max_range) - min_range) / scale) * scale + min_range
        self.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))

    def test_uniform_inferable_quantizer_zero_not_in_range(self):
        min_range = np.asarray([3])
        max_range = np.asarray([10])
        num_bits = 2
        quantizer = qi.pytorch_inferable_quantizers.ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                                                        min_range=min_range,
                                                                                        max_range=max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) <= max_range[0], f'Quantized values should not contain values greater than maximal ' \
                                               f'threshold '
        assert torch.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'
        # Expect to have no more than 2**num_bits unique values
        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')

        # Assert that quantization range was fixed to include 0
        self.assertTrue(0 in quantized_tensor.unique(),
                        f'Expected to find 0 in quantized values but unique values are {quantized_tensor.unique()}')

        # Assert manually quantized values are the same:
        max_range = max_range[0] if max_range[0] >= 0 else 0
        min_range = min_range[0] if min_range[0] <= 0 else 0
        scale = (max_range - min_range) / (2 ** num_bits - 1)

        manually_quantized_tensor = torch.round((torch.clip(input_tensor.to(get_working_device()), min_range,
                                                            max_range) - min_range) / scale) * scale + min_range
        self.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


class TestActivationLUTPOTQuantizer(unittest.TestCase):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                               cluster_centers=np.asarray([25., 85.]),
                                                                               # Not POT threshold
                                                                               threshold=np.asarray([3]),
                                                                               signed=True)
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                               cluster_centers=np.asarray([25., 85.]),
                                                                               # Not float
                                                                               threshold=4,
                                                                               signed=True)

        self.assertEqual('Threshold is expected to be numpy array, but is of type <class \'int\'>', str(e.exception))

    def test_illegal_signed_cluster_centers_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                               cluster_centers=np.asarray([-25., 85.]),
                                                                               threshold=np.asarray([2.]),
                                                                               signed=False)
        self.assertEqual('Expected unsigned cluster centers in unsigned activation quantization', str(e.exception))

    def test_illegal_num_of_cluster_centers_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25, 3, 19, 55])
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=2,
                                                                               cluster_centers=cluster_centers,
                                                                               threshold=np.asarray([2.]),
                                                                               signed=False)
        self.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                         f'{len(cluster_centers)}', str(e.exception))

    def test_illegal_cluster_centers_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                               cluster_centers=np.asarray([-25.6, 25]),
                                                                               threshold=np.asarray([2.]),
                                                                               signed=False)
        self.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def test_illegal_cluster_centers_signed_range_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                               cluster_centers=np.asarray([-25, 25]),
                                                                               threshold=np.asarray([2.]),
                                                                               signed=True,
                                                                               multiplier_n_bits=5)
        self.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def test_illegal_cluster_centers_unsigned_range_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                               cluster_centers=np.asarray([15, 25]),
                                                                               threshold=np.asarray([2.]),
                                                                               signed=False,
                                                                               multiplier_n_bits=4)
        self.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def test_illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=10,
                                                                               cluster_centers=np.asarray([-25, 25]),
                                                                               threshold=np.asarray([2.]),
                                                                               signed=True,
                                                                               multiplier_n_bits=8)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8', str(e.exception))

    def test_warning_num_bit_equal_multiplier_n_bits_inferable_quantizer(self):
        with warnings.catch_warnings(record=True) as w:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                               cluster_centers=np.asarray([-25, 25]),
                                                                               threshold=np.asarray([2.]),
                                                                               signed=True,
                                                                               multiplier_n_bits=8)
        self.assertTrue('Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
                        'in that case' in [str(warning.message) for warning in w])

    def test_illegal_threshold_type_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                               cluster_centers=np.asarray([-25, 25]),
                                                                               threshold=[2.],
                                                                               signed=True,
                                                                               multiplier_n_bits=8)
        self.assertEqual('Threshold is expected to be numpy array, but is of type <class \'list\'>', str(e.exception))

    def test_illegal_threshold_shape_inferable_quantizer(self):
        threshold = np.array([[4., 2.], [2., 16.]])
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                               cluster_centers=np.asarray([-25, 25]),
                                                                               threshold=threshold,
                                                                               signed=True,
                                                                               multiplier_n_bits=8)
        self.assertEqual(f'Threshold is expected to be flatten, but of shape {threshold.shape}', str(e.exception))

    def test_illegal_num_of_thresholds_inferable_quantizer(self):
        threshold = np.asarray([1., 4.])
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                               cluster_centers=np.asarray([-25, 25]),
                                                                               threshold=threshold,
                                                                               signed=True,
                                                                               multiplier_n_bits=8)
        self.assertEqual('For activation, quantization per channel is not supported and threshold should be of length '
                         '1 but is 2', str(e.exception))

    def test_lut_pot_signed_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25])
        thresholds = np.asarray([4.])
        num_bits = 3
        signed = True
        multiplier_n_bits = 8

        quantizer = qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                                                       cluster_centers=cluster_centers,
                                                                                       signed=signed,
                                                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        quant_tensor_values = (cluster_centers / (2 ** (multiplier_n_bits - int(signed)))) * thresholds

        self.assertTrue(len(torch.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(torch.unique(fake_quantized_tensor))} unique values')

        self.assertTrue(np.all(torch.unique(fake_quantized_tensor).detach().cpu().numpy() ==
                               np.sort(quant_tensor_values)))

        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

    def test_lut_pot_unsigned_inferable_quantizer(self):
        cluster_centers = np.asarray([25, 85])
        thresholds = np.asarray([4.])
        num_bits = 3
        signed = False
        multiplier_n_bits = 7

        quantizer = qi.pytorch_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                                                       cluster_centers=cluster_centers,
                                                                                       signed=signed,
                                                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        quant_tensor_values = (cluster_centers / (2 ** (multiplier_n_bits - int(signed)))) * thresholds

        self.assertTrue(len(torch.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(torch.unique(fake_quantized_tensor))} unique values')

        self.assertTrue(np.all(torch.unique(fake_quantized_tensor).detach().cpu().numpy() ==
                               np.sort(quant_tensor_values)))

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(torch.all(fake_quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {fake_quantized_tensor}')
