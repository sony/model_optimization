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
import tensorflow as tf

from model_compression_toolkit import quantizers_infrastructure as qi


class TestKerasWeightsSymmetricQuantizer(unittest.TestCase):

    def test_illegal_num_of_thresholds_symmetric_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                             per_channel=False,
                                                                             threshold=[3., 2.],
                                                                             channel_axis=None,
                                                                             input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_symmetric_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                             per_channel=True,
                                                                             threshold=np.asarray([3., 2.]),
                                                                             channel_axis=None,
                                                                             input_rank=4)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_missing_channel_axis_symmetric_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                             per_channel=True,
                                                                             threshold=[3.,2.],
                                                                             input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                             per_channel=True,
                                                                             threshold=[3.,2.],
                                                                             channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_weights_symmetric_signed_per_tensor_inferable_quantizer(self):
        num_bits = 3
        thresholds = [4.]
        quantizer = qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                                                     per_channel=False,
                                                                                     threshold=thresholds,
                                                                                     channel_axis=None)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['per_channel'] is False)
        self.assertTrue(quantizer_config['channel_axis'] is None)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, dtype=tf.float32)
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be between -4 and 4
        self.assertTrue(np.max(
            quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than maximal '
                                               f'threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= -thresholds[0], f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds[0] / (2 ** (num_bits - 1))
        manually_quantized_tensor = tf.clip_by_value(np.round(input_tensor / scale), clip_value_min=-thresholds[0],
                                                     clip_value_max=thresholds[0] - scale)
        self.assertTrue(np.all(manually_quantized_tensor.numpy() == quantized_tensor.numpy()))

    def test_weights_symmetric_signed_per_channel_inferable_quantizer(self):
        thresholds = [3.,6.,2.]
        num_bits = 2
        quantizer = qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                                                     per_channel=True,
                                                                                     threshold=thresholds,
                                                                                     channel_axis=3,
                                                                                     input_rank=4)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['threshold'] == thresholds))
        self.assertTrue(quantizer_config['per_channel'] is True)
        self.assertTrue(quantizer_config['channel_axis'] == 3)

        thresholds = np.asarray(thresholds)
        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, dtype=tf.float32)
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        # We expect each channel values to be between -threshold to threshold since it's a signed quantization
        for i in range(len(thresholds)):
            channel_slice_i = quantized_tensor[:, :, :, i]
            self.assertTrue(np.max(
                channel_slice_i) < thresholds[
                                i], f'Quantized values should not contain values greater than threshold')
            self.assertTrue(np.min(
                channel_slice_i) >= -thresholds[
                i], f'Quantized values should not contain values lower than threshold')

            self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(channel_slice_i))} unique values')

            # Assert some values are negative (signed quantization)
            self.assertTrue(np.any(channel_slice_i < 0),
                            f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

        # Assert manually quantized values are the same:
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=-thresholds,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.assertTrue(np.all(manually_quantized_tensor == quantized_tensor.numpy()))


class TestKerasWeightsPOTQuantizer(unittest.TestCase):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                       per_channel=False,
                                                                       threshold=[3.],
                                                                       channel_axis=None,
                                                                       input_rank=4)
        self.assertEqual('Expected threshold to be power of 2 but is [3.]', str(e.exception))

    def test_illegal_num_of_thresholds_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                       per_channel=False,
                                                                       threshold=[3.0, 2.0],
                                                                       channel_axis=None,
                                                                       input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                       per_channel=True,
                                                                       threshold=np.asarray([3.0, 2.0]),
                                                                       channel_axis=None,
                                                                       input_rank=4)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_missing_channel_axis_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                       per_channel=True,
                                                                       threshold=[3., 2.],
                                                                       input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                       per_channel=True,
                                                                       threshold=[3., 2.],
                                                                       channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_pot_signed_per_channel_inferable_quantizer(self):
        thresholds = [2., 4., 1.]
        num_bits = 3
        quantizer = qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                                               per_channel=True,
                                                                               threshold=thresholds,
                                                                               channel_axis=-1,
                                                                               input_rank=4)
        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['threshold'] == thresholds))
        self.assertTrue(quantizer_config['per_channel'] is True)
        self.assertTrue(quantizer_config['channel_axis'] == -1)

        thresholds = np.asarray(thresholds)
        delta = thresholds - quantizer.max_range
        is_pot_delta = np.all(
            np.log2(delta) == np.log2(delta).astype(int))
        self.assertTrue(is_pot_delta, f'Expected delta to be POT but: {delta}')

        self.assertTrue(np.all(quantizer.min_range == -1 * thresholds))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, dtype=tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        # We expect each channel values to be between -threshold to threshold since it's a signed quantization
        for i in range(len(thresholds)):
            channel_slice_i = fake_quantized_tensor[:, :, :, i]
            self.assertTrue(np.max(
                channel_slice_i) < thresholds[
                                i], f'Quantized values should not contain values greater than threshold')
            self.assertTrue(np.min(
                channel_slice_i) >= -thresholds[
                i], f'Quantized values should not contain values lower than threshold')
            self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(channel_slice_i))} unique values')

            # Assert some values are negative (signed quantization)
            self.assertTrue(np.any(channel_slice_i < 0),
                            f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

        # Assert manually quantized values are the same:
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=-thresholds,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))

    def test_pot_signed_per_tensor_inferable_quantizer(self):
        thresholds = [1.]
        num_bits = 2
        quantizer = qi.keras_inferable_quantizers.WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                                               per_channel=False,
                                                                               threshold=thresholds,
                                                                               channel_axis=None)
        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['per_channel'] is False)
        self.assertTrue(quantizer_config['channel_axis'] is None)

        thresholds = np.asarray(thresholds)

        delta = thresholds - quantizer.max_range
        is_pot_delta = np.all(
            np.log2(delta) == np.log2(delta).astype(int))
        self.assertTrue(is_pot_delta, f'Expected delta to be POT but: {delta}')

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, dtype=tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        self.assertTrue(np.max(fake_quantized_tensor) < thresholds[
            0], f'Quantized values should not contain values greater than threshold')
        self.assertTrue(np.min(fake_quantized_tensor) >= -thresholds[
            0], f'Quantized values should not contain values lower than threshold')

        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds / (2 ** (num_bits - 1))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=-thresholds,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))


class TestKerasWeightsUniformQuantizer(unittest.TestCase):

    def test_illegal_num_of_minmax_uniform_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                           per_channel=False,
                                                                           min_range=[3., 2.],
                                                                           max_range=[4.,3.],
                                                                           channel_axis=None,
                                                                           input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                           per_channel=True,
                                                                           min_range=[3., 2.],
                                                                           max_range=4,
                                                                           channel_axis=None,
                                                                           input_rank=4)
        self.assertEqual('Expected max_range to be of type list but is <class \'int\'>', str(e.exception))

    def test_missing_channel_axis_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                           per_channel=True,
                                                                           min_range=[3.0, 2.0],
                                                                           max_range=[4.0, 3.0],
                                                                           input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                           per_channel=True,
                                                                           min_range=[3., 2.],
                                                                           max_range=[4., 3.],
                                                                           channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_uniform_per_channel_inferable_quantizer(self):
        num_bits = 3
        min_range = [-10., -3., -8., 0.]
        max_range = [4., 4., 20., 7.]
        channel_axis = 1
        quantizer = qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                                                   per_channel=True,
                                                                                   min_range=min_range,
                                                                                   max_range=max_range,
                                                                                   channel_axis=channel_axis,
                                                                                   input_rank=4)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['max_range'] == max_range))
        self.assertTrue(np.all(quantizer_config['min_range'] == min_range))
        self.assertTrue(quantizer_config['per_channel'] is True)
        self.assertTrue(quantizer_config['channel_axis'] == channel_axis)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 4, 50, 50) * 100 - 50, dtype=tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        min_range = np.asarray(min_range)
        max_range = np.asarray(max_range)

        # We expect each channel values to be between min_range to max_range for each channel
        for i in range(len(min_range)):
            expected_min_channel, expected_max_channel = min_range.flatten()[i], max_range.flatten()[i]
            channel_slice_i = fake_quantized_tensor[:, i, :, :]
            self.assertTrue(np.max(
                channel_slice_i) <= expected_max_channel, f'Quantized values should not contain values greater than '
                                                          f'threshold')
            self.assertTrue(np.min(
                channel_slice_i) >= expected_min_channel, f'Quantized values should not contain values lower than '
                                                          f'threshold')

            self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(channel_slice_i))} unique values')

            # Assert some values are negative (if min range is negative)
            if expected_min_channel < 0:
                self.assertTrue(np.any(channel_slice_i < 0),
                                f'Expected some values to be negative but quantized tensor is {channel_slice_i}')

        # Assert manually quantized values are the same:
        min_range = min_range.reshape((1, 4, 1, 1))
        max_range = max_range.reshape((1, 4, 1, 1))
        scale = (max_range - min_range) / (2 ** num_bits - 1)
        manually_quantized_tensor = np.round((tf.clip_by_value(
            input_tensor, clip_value_min=min_range, clip_value_max=max_range) - min_range) / scale) * scale + min_range
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))

    def test_uniform_per_tensor_inferable_quantizer(self):
        num_bits = 3
        min_range = [-10.]
        max_range = [4.]
        channel_axis = 0

        quantizer = qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                                                   per_channel=False,
                                                                                   min_range=min_range,
                                                                                   max_range=max_range,
                                                                                   channel_axis=channel_axis)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['max_range'] == max_range))
        self.assertTrue(np.all(quantizer_config['min_range'] == min_range))
        self.assertTrue(quantizer_config['per_channel'] is False)
        self.assertTrue(quantizer_config['channel_axis'] == channel_axis)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 4, 50) * 100 - 50, dtype=tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        min_range = np.asarray(min_range)
        max_range = np.asarray(max_range)

        # We expect tensor values values to be between min_range to max_range
        self.assertTrue(np.max(fake_quantized_tensor) <= max_range[
            0], f'Quantized values should not contain values greater than threshold')
        self.assertTrue(np.min(fake_quantized_tensor) >= min_range[
            0], f'Quantized values should not contain values lower than threshold')

        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        # Assert some values are negative
        self.assertTrue(np.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = (max_range - min_range) / (2 ** num_bits - 1)
        manually_quantized_tensor = np.round((tf.clip_by_value(
            input_tensor, clip_value_min=min_range, clip_value_max=max_range) - min_range) / scale) * scale + min_range
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))

    def test_uniform_inferable_quantizer_zero_not_in_range(self):
        num_bits = 3
        min_range = [-10.7, -2.3, -6.6, 0.]
        max_range = [4.1, 4.7, 20.0, 7.0]
        channel_axis = 2

        quantizer = qi.keras_inferable_quantizers.WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                                                   per_channel=True,
                                                                                   min_range=min_range,
                                                                                   max_range=max_range,
                                                                                   channel_axis=channel_axis,
                                                                                   input_rank=4)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['per_channel'] is True)
        self.assertTrue(quantizer_config['channel_axis'] == channel_axis)

        # TODO: check if needed
        # Compute expected adjusted min/max (based on https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_vars):
        # for i, (_min,_max) in enumerate(zip(min_range, max_range)):
        #     if 0 < _min < _max:
        #         min_adj, max_adj = 0, _max-_min
        #     elif _min < _max < 0:
        #         min_adj, max_adj = _min-_max, 0
        #     elif _min<=0<=_max:
        #         _scale = (_max-_min) / (2**num_bits - 1)
        #         min_adj = _scale * round(_min / _scale)
        #         max_adj = _max + min_adj - _min
        #     else:
        #         raise Exception
        #     self.assertTrue(quantizer_config['max_range'][i] == max_adj)
        #     self.assertTrue(quantizer_config['min_range'][i] == min_adj)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 4, 50) * 100 - 50, dtype=tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        # We expect each channel values to be between min_range to max_range for each channel
        for i in range(len(min_range)):
            channel_slice_i = fake_quantized_tensor[:, :, i, :]
            self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(channel_slice_i))} unique values')
            self.assertTrue(0 in np.unique(channel_slice_i),
                            f'zero should be in quantization range, but quantized values are in set: '
                            f'{np.unique(channel_slice_i)}')
