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


class TestWeightsSymmetricQuantizer(unittest.TestCase):

    def test_weights_symmetric_per_tensor_inferable_quantizer(self):
        quantizer = qi.pytorch_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=3,
                                                                                       per_channel=False,
                                                                                       threshold=np.asarray([4]))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < 4, f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= -4, f'Quantized values should not contain values lower than minimal threshold'

        self.assertTrue(len(quantized_tensor.unique()) <= 2 ** 3,
                        f'Quantized tensor expected to have no more than {2 ** 3} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_weights_symmetric_per_channel_inferable_quantizer(self):
        thresholds = [3, 6, 2]
        quantizer = qi.pytorch_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=2,
                                                                                       per_channel=True,
                                                                                       threshold=np.asarray(thresholds),
                                                                                       channel_axis=3)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))
        fake_quantized_tensor = quantized_tensor.dequantize()

        # We expect each channel values to be between -threshold to threshold since it's a signed quantization
        for i in range(len(thresholds)):
            channel_slice_i = fake_quantized_tensor[:, :, :, i]
            assert torch.max(
                channel_slice_i) < thresholds[
                       i], f'Quantized values should not contain values greater than threshold'
            assert torch.min(
                channel_slice_i) >= -thresholds[
                i], f'Quantized values should not contain values lower than threshold'
            self.assertTrue(len(channel_slice_i.unique()) <= 2 ** 8,
                            f'Quantized tensor expected to have no more than {2 ** 8} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')
            # Assert some values are negative (signed quantization)
            self.assertTrue(torch.any(channel_slice_i < 0),
                            f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

    def test_weights_symmetric_per_channel_no_axis(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                               per_channel=True,
                                                                               threshold=np.asarray([1]))
        self.assertEqual('Channel axis is missing in per channel quantization ', str(e.exception))


class TestWeightsPOTQuantizer(unittest.TestCase):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                         per_channel=False,
                                                                         # Not POT threshold
                                                                         threshold=np.asarray([3]))
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.assertRaises(Exception) as e:
            qi.pytorch_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                         per_channel=False,
                                                                         # Not POT threshold
                                                                         threshold=np.asarray([2, 3]))
        self.assertEqual('Expected threshold to be power of 2 but is [2 3]', str(e.exception))

    def test_pot_per_channel_inferable_quantizer(self):
        thresholds = [2, 4, 1]
        quantizer = qi.pytorch_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=3,
                                                                                 per_channel=True,
                                                                                 threshold=np.asarray(thresholds),
                                                                                 channel_axis=3)

        is_pot_scales = torch.all(
            quantizer.scales.log2().int() == quantizer.scales.log2())
        self.assertTrue(is_pot_scales, f'Expected scales to be POT but: {quantizer.scales}')

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # We expect each channel values to be between -threshold to threshold since it's a signed quantization
        for i in range(len(thresholds)):
            channel_slice_i = fake_quantized_tensor[:, :, :, i]
            assert torch.max(
                channel_slice_i) < thresholds[
                       i], f'Quantized values should not contain values greater than threshold'
            assert torch.min(
                channel_slice_i) >= -thresholds[
                i], f'Quantized values should not contain values lower than threshold'
            self.assertTrue(len(channel_slice_i.unique()) <= 2 ** 3,
                            f'Quantized tensor expected to have no more than {2 ** 3} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')
            # Assert some values are negative (signed quantization)
            self.assertTrue(torch.any(channel_slice_i < 0),
                            f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

    def test_pot_per_tensor_inferable_quantizer(self):
        thresholds = [1]
        quantizer = qi.pytorch_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=2,
                                                                                 per_channel=False,
                                                                                 threshold=np.asarray(thresholds))
        is_pot_scales = torch.all(
            quantizer.scales.log2().int() == quantizer.scales.log2())
        self.assertTrue(is_pot_scales, f'Expected scales to be POT but: {quantizer.scales}')
        self.assertTrue(len(quantizer.scales) == 1,
                        f'Expected to have one scale in per-tensor quantization but found '
                        f'{len(quantizer.scales)} scales')

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        assert torch.max(fake_quantized_tensor) < thresholds[
            0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(fake_quantized_tensor) >= -thresholds[
            0], f'Quantized values should not contain values lower than threshold'
        self.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** 2,
                        f'Quantized tensor expected to have no more than {2 ** 2} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')
