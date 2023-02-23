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
# ==============================================================================g
import numpy as np
import torch

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizer_utils import \
    get_working_device
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers import \
    WeightsUniformInferableQuantizer, WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest


class TestPytorchWeightsSymmetricInferablePerTensorQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        num_bits = 3
        thresholds = np.asarray([4])
        quantizer = WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                       per_channel=False,
                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
        assert torch.max(
            quantized_tensor) < thresholds[
                   0], f'Quantized values should not contain values greater than maximal threshold'
        assert torch.min(
            quantized_tensor) >= -thresholds[
            0], f'Quantized values should not contain values lower than minimal threshold'

        self.unit_test.assertTrue(len(quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(torch.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds[0] / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.clip(torch.round(input_tensor.to(get_working_device()) / scale),
                                               -thresholds[0], thresholds[0] - scale)
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


class TestPytorchWeightsSymmetricInferablePerChannelQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        thresholds = np.asarray([3, 6, 2])
        num_bits = 2
        quantizer = WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                       per_channel=True,
                                                       threshold=thresholds,
                                                       channel_axis=3)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor.to(get_working_device()))
        fake_quantized_tensor = quantized_tensor

        # We expect each channel values to be between -threshold to threshold since it's a signed quantization
        for i in range(len(thresholds)):
            channel_slice_i = fake_quantized_tensor[:, :, :, i]
            assert torch.max(
                channel_slice_i) < thresholds[
                       i], f'Quantized values should not contain values greater than threshold'
            assert torch.min(
                channel_slice_i) >= -thresholds[
                i], f'Quantized values should not contain values lower than threshold'
            self.unit_test.assertTrue(len(channel_slice_i.unique()) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')
            # Assert some values are negative (signed quantization)
            self.unit_test.assertTrue(torch.any(channel_slice_i < 0),
                            f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

        # Assert manually quantized values are the same:
        thresholds = torch.Tensor(thresholds).to(get_working_device())
        thresholds = thresholds.reshape((1, 1, 1, 3))
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == quantized_tensor))


class TestPytorchWeightsSymmetricInferableNoAxisQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=np.asarray([1]))
        self.unit_test.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))


class TestPytorchWeightsPOTInferableQuantizerRaise(BaseInferableQuantizerTest):

    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         # Not POT threshold
                                         threshold=np.asarray([3]))
        self.unit_test.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         # More than one threshold in per-tensor quantization
                                         threshold=np.asarray([2, 3]))
        self.unit_test.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2', str(e.exception))


class TestPytorchWeightsPOTInferablePerChannelQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        thresholds = np.asarray([2, 4, 1])
        num_bits = 3
        quantizer = WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                 per_channel=True,
                                                 threshold=thresholds,
                                                 channel_axis=3)

        is_pot_scales = torch.all(
            quantizer.scales.log2().int() == quantizer.scales.log2())
        self.unit_test.assertTrue(is_pot_scales, f'Expected scales to be POT but: {quantizer.scales}')

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
            self.unit_test.assertTrue(len(channel_slice_i.unique()) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')
            # Assert some values are negative (signed quantization)
            self.unit_test.assertTrue(torch.any(channel_slice_i < 0),
                            f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

        # Assert manually quantized values are the same:
        thresholds = torch.Tensor(thresholds).to(get_working_device())
        thresholds = thresholds.reshape((1, 1, 1, 3))
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))

    def test_pot_per_tensor_inferable_quantizer(self):
        thresholds = np.asarray([1])
        num_bits = 2
        quantizer = WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                 per_channel=False,
                                                 threshold=thresholds)
        is_pot_scales = torch.all(
            quantizer.scales.log2().int() == quantizer.scales.log2())
        self.unit_test.assertTrue(is_pot_scales, f'Expected scales to be POT but: {quantizer.scales}')
        self.unit_test.assertTrue(len(quantizer.scales) == 1,
                        f'Expected to have one scale in per-tensor quantization but found '
                        f'{len(quantizer.scales)} scales')

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 50, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        assert torch.max(fake_quantized_tensor) < thresholds[
            0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(fake_quantized_tensor) >= -thresholds[
            0], f'Quantized values should not contain values lower than threshold'
        self.unit_test.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')
        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        thresholds = torch.Tensor(thresholds).to(get_working_device())
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = torch.round(
            torch.clip(input_tensor.to(get_working_device()), -thresholds, thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))


class TestPytorchWeightsUniformInferablePerChannelQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        num_bits = 3
        min_range = np.asarray([-10, -3, -8, 0])
        max_range = np.asarray([4, 4, 20, 7])
        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                     per_channel=True,
                                                     min_range=min_range,
                                                     max_range=max_range,
                                                     channel_axis=2)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 4, 50) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # We expect each channel values to be between min_range to max_range for each channel
        for i in range(len(min_range)):
            expected_min_channel, expected_max_channel = min_range[i], max_range[i]
            channel_slice_i = fake_quantized_tensor[:, :, i, :]
            assert torch.max(
                channel_slice_i) <= expected_max_channel, f'Quantized values should not contain values greater than ' \
                                                          f'threshold'
            assert torch.min(
                channel_slice_i) >= expected_min_channel, f'Quantized values should not contain values lower than ' \
                                                          f'threshold'
            self.unit_test.assertTrue(len(channel_slice_i.unique()) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')

            # Assert some values are negative (if min range is negative)
            if expected_min_channel < 0:
                self.unit_test.assertTrue(torch.any(channel_slice_i < 0),
                                f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

        # Assert manually quantized values are the same:
        min_range = torch.reshape(torch.Tensor(min_range).to(get_working_device()), (1, 1, 4, 1))
        max_range = torch.reshape(torch.Tensor(max_range).to(get_working_device()), (1, 1, 4, 1))
        scale = (max_range - min_range) / (2 ** num_bits - 1)
        manually_quantized_tensor = torch.round((torch.clip(input_tensor.to(get_working_device()), min_range,
                                                            max_range) - min_range) / scale) * scale + min_range
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))


class TestPytorchWeightsUniformInferablePerTensorQuantizer(BaseInferableQuantizerTest):
    
    def run_test(self):
        num_bits = 3
        min_range = np.asarray([-10])
        max_range = np.asarray([4])
        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                     per_channel=False,
                                                     min_range=min_range,
                                                     max_range=max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 4, 50) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # We expect tensor values values to be between min_range to max_range
        assert torch.max(fake_quantized_tensor) <= max_range[
            0], f'Quantized values should not contain values greater than threshold'
        assert torch.min(fake_quantized_tensor) >= min_range[
            0], f'Quantized values should not contain values lower than threshold'
        self.unit_test.assertTrue(len(fake_quantized_tensor.unique()) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(fake_quantized_tensor.unique())} unique values')

        # Assert some values are negative
        self.unit_test.assertTrue(torch.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        min_range = torch.Tensor(min_range).to(get_working_device())
        max_range = torch.Tensor(max_range).to(get_working_device())
        scale = (max_range - min_range) / (2 ** num_bits - 1)
        manually_quantized_tensor = torch.round((torch.clip(input_tensor.to(get_working_device()), min_range,
                                                            max_range) - min_range) / scale) * scale + min_range
        self.unit_test.assertTrue(torch.all(manually_quantized_tensor == fake_quantized_tensor))


class TestPytorchWeightsUniformInferableQuantizerZeroNotInRange(BaseInferableQuantizerTest):
    
    def run_test(self):
        num_bits = 3
        min_range = np.asarray([-10.7, 2.3, -6.6, 0])
        max_range = np.asarray([-4.1, 4.7, 20, 7])
        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                     per_channel=True,
                                                     min_range=min_range,
                                                     max_range=max_range,
                                                     channel_axis=2)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 4, 50) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # We expect each channel values to be between min_range to max_range for each channel
        for i in range(len(min_range)):
            expected_min_channel, expected_max_channel = min_range[i], max_range[i]
            channel_slice_i = fake_quantized_tensor[:, :, i, :]
            self.unit_test.assertTrue(len(channel_slice_i.unique()) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')
            self.unit_test.assertTrue(0 in channel_slice_i.unique(),
                            f'zero should be in quantization range, but quantized values are in set: {channel_slice_i.unique()}')
