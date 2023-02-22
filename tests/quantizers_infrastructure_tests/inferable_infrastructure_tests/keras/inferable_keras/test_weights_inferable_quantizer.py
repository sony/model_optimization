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
import tensorflow as tf

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers import \
    WeightsPOTInferableQuantizer, WeightsUniformInferableQuantizer, WeightsSymmetricInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.weights_inferable_quantizers.weights_lut_pot_inferable_quantizer import \
    WeightsLUTPOTInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer


class TestKerasWeightsSymmetricQuantizer(unittest.TestCase):

    def test_illegal_num_of_thresholds_symmetric_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=False,
                                               threshold=[3., 2.],
                                               channel_axis=None,
                                               input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_symmetric_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=np.asarray([3., 2.]),
                                               channel_axis=None,
                                               input_rank=4)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_missing_channel_axis_symmetric_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=[3., 2.],
                                               input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=[3., 2.],
                                               channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_weights_symmetric_signed_per_tensor_inferable_quantizer(self):
        num_bits = 3
        thresholds = [4.]
        quantizer = WeightsSymmetricInferableQuantizer(num_bits=num_bits,
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
        thresholds = [3., 6., 2.]
        num_bits = 2
        quantizer = WeightsSymmetricInferableQuantizer(num_bits=num_bits,
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
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         threshold=[3.],
                                         channel_axis=None,
                                         input_rank=4)
        self.assertEqual('Expected threshold to be power of 2 but is [3.]', str(e.exception))

    def test_illegal_num_of_thresholds_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         threshold=[3.0, 2.0],
                                         channel_axis=None,
                                         input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=np.asarray([3.0, 2.0]),
                                         channel_axis=None,
                                         input_rank=4)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_missing_channel_axis_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=[3., 2.],
                                         input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=[3., 2.],
                                         channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_pot_signed_per_channel_inferable_quantizer(self):
        thresholds = [2., 4., 1.]
        num_bits = 3
        quantizer = WeightsPOTInferableQuantizer(num_bits=num_bits,
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
        quantizer = WeightsPOTInferableQuantizer(num_bits=num_bits,
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
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=False,
                                             min_range=[3., 2.],
                                             max_range=[4., 3.],
                                             channel_axis=None,
                                             input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=True,
                                             min_range=[3., 2.],
                                             max_range=4,
                                             channel_axis=None,
                                             input_rank=4)
        self.assertEqual('Expected max_range to be of type list but is <class \'int\'>', str(e.exception))

    def test_missing_channel_axis_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=True,
                                             min_range=[3.0, 2.0],
                                             max_range=[4.0, 3.0],
                                             input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
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
        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
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

        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
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

        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
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


class TestKerasWeightsLUTSymmetricQuantizer(unittest.TestCase):

    def illegal_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                         per_channel, input_rank, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def illegal_num_of_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                                per_channel, input_rank, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=2,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                         f'{len(cluster_centers)}', str(e.exception))

    def illegal_cluster_centers_range_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                               per_channel, input_rank, channel_axis,
                                                               multiplier_n_bits):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits,
                                input_rank=input_rank)
        self.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                               cluster_centers,
                                                                               num_bits, input_rank,
                                                                               per_channel, channel_axis,
                                                                               multiplier_n_bits):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits,
                                input_rank=input_rank)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
                         , str(e.exception))

    def warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                         cluster_centers,
                                                                         num_bits, input_rank,
                                                                         per_channel, channel_axis,
                                                                         multiplier_n_bits):
        with warnings.catch_warnings(record=True) as w:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits,
                                input_rank=input_rank)
        self.assertTrue('Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
                        'in that case, consider using SymmetricInferableQuantizer instead'
                        in [str(warning.message) for warning in w])

    def illegal_num_of_thresholds_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                           per_channel, channel_axis, input_rank):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2', str(e.exception))

    def illegal_threshold_type_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                        per_channel, channel_axis, input_rank):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def missing_channel_axis_inferable_quantizer(self, inferable_quantizer, threshold, cluster_centers,
                                                 per_channel, input_rank):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                input_rank=input_rank)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def missing_input_rank_inferable_quantizer(self, inferable_quantizer, threshold, cluster_centers,
                                               per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def weights_inferable_quantizer_test(self, inferable_quantizer, num_bits, threshold, cluster_centers,
                                         per_channel, channel_axis, input_rank, multiplier_n_bits, eps):
        quantizer = inferable_quantizer(num_bits=num_bits,
                                        per_channel=per_channel,
                                        cluster_centers=cluster_centers,
                                        threshold=threshold,
                                        channel_axis=channel_axis,
                                        input_rank=input_rank,
                                        multiplier_n_bits=multiplier_n_bits,
                                        eps=eps)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['threshold'] == np.asarray(threshold)))
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['per_channel'] == per_channel)
        self.assertTrue(quantizer_config['channel_axis'] == channel_axis)
        self.assertTrue(quantizer_config['input_rank'] == input_rank)
        self.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.assertTrue(quantizer_config['eps'] == eps)

        # test permute
        perm_vec = list(np.arange(input_rank))
        if per_channel and channel_axis not in [-1, input_rank - 1]:
            perm_vec[channel_axis] = input_rank - 1
            perm_vec[input_rank - 1] = channel_axis

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, dtype=tf.float32)

        # change the input only when channel_axis is not the last axis
        input_tensor = tf.transpose(input_tensor, perm=perm_vec)

        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        self.assertTrue(quantized_tensor.shape == input_tensor.shape, f'Quantized tensor should be in the same shape '
                                                                      f'as the input tensor')

        # return the output's channel axis to the last axis
        # change the input only when channel_axis is not the last axis
        quantized_tensor = tf.transpose(quantized_tensor, perm=perm_vec)

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(threshold))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - 1))

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (multiplier_n_bits - 1) - 1
        clip_min = -2 ** (multiplier_n_bits - 1)

        if per_channel:
            for i in range(len(threshold)):
                channel_slice_i = quantized_tensor[:, :, :, i]
                channel_quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold[i]
                self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                                f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                f'{len(np.unique(channel_slice_i))} unique values')
                self.assertTrue(np.all(np.unique(channel_slice_i) == np.sort(channel_quant_tensor_values)))

                # Check quantized tensor assigned correctly
                tensor = tf.clip_by_value((input_tensor / (threshold[i] + eps)) * (2 ** (num_bits - 1)),
                                          clip_value_max=clip_max, clip_value_min=clip_min)
                tensor = tf.expand_dims(tf.transpose(tensor, perm=perm_vec)[:, :, :, i], -1)
                expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
                cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
                centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

                self.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[i] == channel_slice_i),
                                "Quantized tensor values weren't assigned correctly")
        else:
            quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold
            self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(quantized_tensor))} unique values')
            self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

            # Check quantized tensor assigned correctly
            tensor = tf.clip_by_value((input_tensor / (threshold[0] + eps)) * (2 ** (num_bits - 1)),
                                      clip_value_max=clip_max, clip_value_min=clip_min)
            tensor = tf.expand_dims(tensor, -1)
            expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
            cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
            centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

            self.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[0] == quantized_tensor),
                            "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_weights_lut_symmetric_inferable_quantizer(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25.6, 25])
        per_channel = False
        channel_axis = None
        input_rank = 4

        threshold = [2.]
        self.illegal_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                              threshold=threshold,
                                                              cluster_centers=cluster_centers, per_channel=per_channel,
                                                              channel_axis=channel_axis,
                                                              input_rank=input_rank)

        cluster_centers = np.asarray([-25, 25, 3, 19, 55])
        self.illegal_num_of_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                     threshold=threshold,
                                                                     cluster_centers=cluster_centers,
                                                                     per_channel=per_channel,
                                                                     channel_axis=channel_axis,
                                                                     input_rank=input_rank)
        multiplier_n_bits = 5
        cluster_centers = np.asarray([-25, 25])
        self.illegal_cluster_centers_range_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                    threshold=threshold,
                                                                    cluster_centers=cluster_centers,
                                                                    per_channel=per_channel,
                                                                    channel_axis=channel_axis,
                                                                    multiplier_n_bits=multiplier_n_bits,
                                                                    input_rank=input_rank)

        multiplier_n_bits = 8
        num_bits = 10
        self.illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits,
            input_rank=input_rank)

        multiplier_n_bits = 8
        num_bits = 8
        self.warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits,
            input_rank=input_rank)

        threshold = np.asarray([3., 2.])
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                             threshold=threshold,
                                                             cluster_centers=cluster_centers, per_channel=per_channel,
                                                             channel_axis=channel_axis, input_rank=input_rank)

        threshold = [2., 7.]
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                threshold=threshold, cluster_centers=cluster_centers,
                                                                per_channel=per_channel, channel_axis=channel_axis,
                                                                input_rank=input_rank)

        threshold = [2.]
        per_channel = True
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=inferable_quantizer,
                                                      threshold=threshold,
                                                      cluster_centers=cluster_centers, per_channel=per_channel,
                                                      input_rank=input_rank)

        self.missing_input_rank_inferable_quantizer(inferable_quantizer=inferable_quantizer,
                                                    threshold=threshold,
                                                    cluster_centers=cluster_centers, per_channel=per_channel,
                                                    channel_axis=channel_axis)

        # test per channel
        threshold = [3., 8., 7.]
        channel_axis = 3
        multiplier_n_bits = 8
        eps = 1e-8
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                              eps=eps)

        # test per channel and channel axis is not last
        threshold = [3., 8., 7.]
        channel_axis = 1
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                              eps=eps)

        # test per tensor
        threshold = [3.]
        channel_axis = None
        per_channel = False
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                              eps=eps)


class TestKerasWeightsLUTPOTQuantizer(TestKerasWeightsLUTSymmetricQuantizer):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsLUTPOTInferableQuantizer(num_bits=8,
                                            per_channel=False,
                                            cluster_centers=np.asarray([25., 85.]),
                                            threshold=[3.],
                                            channel_axis=None,
                                            input_rank=4)
        self.assertEqual('Expected threshold to be power of 2 but is 3.0', str(e.exception))

    def test_weights_lut_pot_inferable_quantizer(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25.6, 25])
        per_channel = False
        channel_axis = None
        input_rank = 4

        threshold = [2.]
        self.illegal_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                              threshold=threshold,
                                                              cluster_centers=cluster_centers, per_channel=per_channel,
                                                              channel_axis=channel_axis,
                                                              input_rank=input_rank)

        cluster_centers = np.asarray([-25, 25, 3, 19, 55])
        self.illegal_num_of_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                     threshold=threshold,
                                                                     cluster_centers=cluster_centers,
                                                                     per_channel=per_channel,
                                                                     channel_axis=channel_axis,
                                                                     input_rank=input_rank)
        multiplier_n_bits = 5
        cluster_centers = np.asarray([-25, 25])
        self.illegal_cluster_centers_range_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                    threshold=threshold,
                                                                    cluster_centers=cluster_centers,
                                                                    per_channel=per_channel,
                                                                    channel_axis=channel_axis,
                                                                    multiplier_n_bits=multiplier_n_bits,
                                                                    input_rank=input_rank)

        multiplier_n_bits = 8
        num_bits = 10
        self.illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits,
            input_rank=input_rank)

        multiplier_n_bits = 8
        num_bits = 8
        self.warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits,
            input_rank=input_rank)

        threshold = np.asarray([2., 16.])
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                             threshold=threshold,
                                                             cluster_centers=cluster_centers, per_channel=per_channel,
                                                             channel_axis=channel_axis, input_rank=input_rank)

        threshold = [2., 8.]
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                threshold=threshold, cluster_centers=cluster_centers,
                                                                per_channel=per_channel, channel_axis=channel_axis,
                                                                input_rank=input_rank)

        threshold = [2.]
        per_channel = True
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=inferable_quantizer,
                                                      threshold=threshold,
                                                      cluster_centers=cluster_centers, per_channel=per_channel,
                                                      input_rank=input_rank)

        self.missing_input_rank_inferable_quantizer(inferable_quantizer=inferable_quantizer,
                                                    threshold=threshold,
                                                    cluster_centers=cluster_centers, per_channel=per_channel,
                                                    channel_axis=channel_axis)

        # test per channel
        threshold = [2., 8., 32.]
        channel_axis = 3
        multiplier_n_bits = 8
        eps = 1e-8
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                              eps=eps)

        # test per channel and channel axis is not last
        threshold = [2., 8., 32.]
        channel_axis = 1
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                              eps=eps)

        # test per tensor
        threshold = [4.]
        channel_axis = None
        per_channel = False
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                              eps=eps)
