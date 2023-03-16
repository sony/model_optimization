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
import warnings

import numpy as np
import torch

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import \
    ActivationLutPOTInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizer_utils import \
    get_working_device
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest


class BasePytorchActivationLUTQuantizerTest(BaseInferableQuantizerTest):

    def test_illegal_pot_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([25., 85.]),
                                               # Not POT threshold
                                               threshold=np.asarray([3]),
                                               signed=True)
        self.unit_test.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([25., 85.]),
                                               # Not float
                                               threshold=4,
                                               signed=True)

        self.unit_test.assertEqual('Threshold is expected to be numpy array, but is of type <class \'int\'>',
                                   str(e.exception))

    def test_illegal_signed_cluster_centers_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([-25., 85.]),
                                               threshold=np.asarray([2.]),
                                               signed=False)
        self.unit_test.assertEqual('Expected unsigned cluster centers in unsigned activation quantization',
                                   str(e.exception))

    def test_illegal_num_of_cluster_centers_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25, 3, 19, 55])
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=2,
                                               cluster_centers=cluster_centers,
                                               threshold=np.asarray([2.]),
                                               signed=False)
        self.unit_test.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                                   f'{len(cluster_centers)}', str(e.exception))

    def test_illegal_cluster_centers_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([-25.6, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=False)
        self.unit_test.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def test_illegal_cluster_centers_signed_range_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([-25, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=True,
                                               multiplier_n_bits=5)
        self.unit_test.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def test_illegal_cluster_centers_unsigned_range_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([15, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=False,
                                               multiplier_n_bits=4)
        self.unit_test.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def test_illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=10,
                                               cluster_centers=np.asarray([-25, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=True,
                                               multiplier_n_bits=8)
        self.unit_test.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8',
                                   str(e.exception))

    def test_warning_num_bit_equal_multiplier_n_bits_inferable_quantizer(self):
        with warnings.catch_warnings(record=True) as w:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               cluster_centers=np.asarray([-25, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=True,
                                               multiplier_n_bits=8)
        self.unit_test.assertTrue(
            'Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
            'in that case, consider using SymmetricInferableQuantizer instead'
            in [str(warning.message) for warning in w])

    def test_illegal_threshold_type_inferable_quantizer(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([-25, 25]),
                                               threshold=[2.],
                                               signed=True,
                                               multiplier_n_bits=8)
        self.unit_test.assertEqual('Threshold is expected to be numpy array, but is of type <class \'list\'>',
                                   str(e.exception))

    def test_illegal_threshold_shape_inferable_quantizer(self):
        threshold = np.array([[4., 2.], [2., 16.]])
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([-25, 25]),
                                               threshold=threshold,
                                               signed=True,
                                               multiplier_n_bits=8)
        self.unit_test.assertEqual(f'Threshold is expected to be flatten, but of shape {threshold.shape}',
                                   str(e.exception))

    def test_illegal_num_of_thresholds_inferable_quantizer(self):
        threshold = np.asarray([1., 4.])
        with self.unit_test.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               cluster_centers=np.asarray([-25, 25]),
                                               threshold=threshold,
                                               signed=True,
                                               multiplier_n_bits=8)
        self.unit_test.assertEqual(
            'For activation, quantization per channel is not supported and threshold should be of length '
            '1 but is 2', str(e.exception))

    def test_lut_pot_signed_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25])
        thresholds = np.asarray([4.])
        num_bits = 3
        signed = True
        multiplier_n_bits = 8

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       cluster_centers=cluster_centers,
                                                       signed=signed,
                                                       multiplier_n_bits=
                                                       multiplier_n_bits,
                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))
        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - int(signed)))

        fake_quantized_tensor = fake_quantized_tensor.detach().cpu().numpy()

        self.unit_test.assertTrue(np.max(
            fake_quantized_tensor) <= (max_threshold - delta_threshold), f'Quantized values should not contain values '
                                                                         f'greater than maximal threshold ')
        self.unit_test.assertTrue(np.min(
            fake_quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                      f'threshold ')

        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')

        quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - int(signed))) * thresholds
        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')
        self.unit_test.assertTrue(np.all(np.unique(fake_quantized_tensor)
                                         == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (multiplier_n_bits - 1) - 1
        clip_min = -2 ** (multiplier_n_bits - 1)

        tensor = torch.clip((input_tensor / thresholds) * (2 ** (multiplier_n_bits - int(signed))),
                            min=clip_min, max=clip_max)
        tensor = tensor.unsqueeze(-1)
        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = torch.argmin(torch.abs(tensor - expanded_cluster_centers), dim=-1)
        centers = cluster_centers.flatten()[cluster_assignments]

        self.unit_test.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - int(signed))) * thresholds ==
                                         fake_quantized_tensor), "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(np.any(fake_quantized_tensor < 0),
                                  f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

    def test_lut_pot_unsigned_inferable_quantizer(self):
        cluster_centers = np.asarray([25, 45])
        thresholds = np.asarray([4.])
        num_bits = 3
        signed = False
        multiplier_n_bits = 7

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       cluster_centers=cluster_centers,
                                                       signed=signed,
                                                       multiplier_n_bits=
                                                       multiplier_n_bits,
                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a unsigned quantization, so we expect all values to be between 0
        # and max(threshold)
        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - int(signed)))

        fake_quantized_tensor = fake_quantized_tensor.detach().cpu().numpy()

        self.unit_test.assertTrue(np.max(
            fake_quantized_tensor) < (max_threshold - delta_threshold), f'Quantized values should not contain values '
                                                                        f'greater than maximal threshold ')
        self.unit_test.assertTrue(np.min(
            fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0 ')

        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')

        quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - int(signed))) * thresholds
        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')

        self.unit_test.assertTrue(np.all(np.unique(fake_quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        clip_max = 2 ** multiplier_n_bits - 1
        clip_min = 0

        tensor = torch.clip((input_tensor / thresholds) * (2 ** (multiplier_n_bits - int(signed))),
                            min=clip_min, max=clip_max)
        tensor = tensor.unsqueeze(-1)
        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = torch.argmin(torch.abs(tensor - expanded_cluster_centers), dim=-1)
        centers = cluster_centers.flatten()[cluster_assignments]

        self.unit_test.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - int(signed))) * thresholds ==
                                         fake_quantized_tensor), "Quantized tensor values weren't assigned correctly")

        # Assert all values are non-negative (unsigned quantization)
        self.unit_test.assertTrue(np.all(fake_quantized_tensor >= 0),
                                  f'Expected all values to be non-negative but quantized tensor is {fake_quantized_tensor}')


class TestPytorchActivationPOTLUTQuantizerAssertions(BasePytorchActivationLUTQuantizerTest):

    def run_test(self):
        self.test_illegal_pot_inferable_quantizer()
        self.test_illegal_cluster_centers_inferable_quantizer()
        self.test_illegal_num_of_cluster_centers_inferable_quantizer()
        self.test_illegal_cluster_centers_signed_range_inferable_quantizer()
        self.test_illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer()
        self.test_warning_num_bit_equal_multiplier_n_bits_inferable_quantizer()
        self.test_illegal_num_of_thresholds_inferable_quantizer()
        self.test_illegal_threshold_type_inferable_quantizer()
        self.test_illegal_signed_cluster_centers_inferable_quantizer()
        self.test_illegal_cluster_centers_unsigned_range_inferable_quantizer()
        self.test_illegal_threshold_shape_inferable_quantizer()


class TestPytorchActivationPOTLUTQuantizer(BasePytorchActivationLUTQuantizerTest):

    def run_test(self):
        self.test_lut_pot_signed_inferable_quantizer()
        self.test_lut_pot_unsigned_inferable_quantizer()
