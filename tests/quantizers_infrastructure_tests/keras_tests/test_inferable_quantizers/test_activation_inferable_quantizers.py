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

from model_compression_toolkit import quantizers_infrastructure as qi


class TestKerasActivationsSymmetricQuantizer(unittest.TestCase):

    def test_activation_signed_symmetric_inferable_quantizer(self):
        num_bits = 3
        thresholds = [4.]
        signed = True

        quantizer = qi.keras_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                                                        threshold=thresholds,
                                                                                        signed=signed)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
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
        scale = thresholds[0] / (2 ** (num_bits - int(signed)))
        manually_quantized_tensor = tf.clip_by_value(np.round(input_tensor / scale), clip_value_min=-thresholds[0],
                                                     clip_value_max=thresholds[0] - scale)
        self.assertTrue(np.all(manually_quantized_tensor.numpy() == quantized_tensor.numpy()))

    def test_activation_unsigned_symmetric_inferable_quantizer(self):
        thresholds = [4.]
        num_bits = 2
        signed = False

        quantizer = qi.keras_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                                                        threshold=thresholds,
                                                                                        signed=signed)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a unsigned quantization, so we expect all values to be between 0 and 4
        self.assertTrue(np.max(
            quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than maximal '
                                               f'threshold')
        self.assertTrue(np.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds[0] / (2 ** num_bits - int(signed))
        manually_quantized_tensor = tf.clip_by_value(np.round(input_tensor / scale), clip_value_min=0,
                                                     clip_value_max=thresholds[0] - scale)
        self.assertTrue(np.all(manually_quantized_tensor.numpy() == quantized_tensor.numpy()))


class TestKerasActivationsPOTQuantizer(unittest.TestCase):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=8,
                                                                          threshold=[3.],
                                                                          signed=True)
        self.assertEqual('Expected threshold to be power of 2 but is [3.]', str(e.exception))

    def test_pot_signed_inferable_quantizer(self):
        thresholds = [1.]
        num_bits = 2
        signed = True

        quantizer = qi.keras_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                                                  signed=signed,
                                                                                  threshold=thresholds)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)

        delta = thresholds - quantizer.max_range
        is_pot_delta = np.all(
            np.log2(delta) == np.log2(delta).astype(int))
        self.assertTrue(is_pot_delta, f'Expected delta to be POT but: {delta}')

        self.assertTrue(np.all(quantizer.min_range == -1 * thresholds))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
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
        scale = thresholds / (2 ** (num_bits - int(signed)))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=-thresholds,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))

    def test_pot_unsigned_inferable_quantizer(self):
        thresholds = [1.]
        num_bits = 2
        signed = False

        quantizer = qi.keras_inferable_quantizers.ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                                                  signed=signed,
                                                                                  threshold=thresholds)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)

        delta = thresholds - quantizer.max_range
        is_pot_delta = np.all(
            np.log2(delta) == np.log2(delta).astype(int))
        self.assertTrue(is_pot_delta, f'Expected delta to be POT but: {delta}')

        self.assertTrue(np.all(quantizer.min_range == 0))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        self.assertTrue(np.max(fake_quantized_tensor) < thresholds[
            0], f'Quantized values should not contain values greater than threshold')

        self.assertTrue(np.min(fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.any(fake_quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds / (2 ** num_bits - int(signed))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=0,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))


class TestKerasActivationsUniformQuantizer(unittest.TestCase):

    def test_uniform_inferable_quantizer(self):
        min_range = [-10.]
        max_range = [5.]
        num_bits = 2
        quantizer = qi.keras_inferable_quantizers.ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                                                      min_range=min_range,
                                                                                      max_range=max_range)

        min_range = np.asarray(min_range)
        max_range = np.asarray(max_range)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(quantizer_config['min_range'] == min_range)
        self.assertTrue(quantizer_config['max_range'] == max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 4, 50) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

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
        manually_quantized_tensor = \
            np.round((tf.clip_by_value(input_tensor, clip_value_min=min_range,
                                       clip_value_max=max_range) - min_range) / scale) * scale + min_range
        self.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))

    def test_uniform_inferable_quantizer_zero_not_in_range(self):
        min_range = [3.]
        max_range = [10.]
        num_bits = 2

        quantizer = qi.keras_inferable_quantizers.ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                                                      min_range=min_range,
                                                                                      max_range=max_range)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        # TODO: fix check
        # self.assertTrue(quantizer_config['min_range'] == min_range)
        # self.assertTrue(quantizer_config['max_range'] == max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 4, 50) * 100 - 50, tf.float32)
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


class TestKerasActivationsLUTPOTQuantizer(unittest.TestCase):

    def test_illegal_pot_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                             cluster_centers=np.asarray([25., 85.]),
                                                                             threshold=[3.],
                                                                             signed=True)
        self.assertEqual('Expected threshold to be power of 2 but is [3.0]', str(e.exception))

    def test_illegal_cluster_centers_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                             cluster_centers=np.asarray([25.9, 85.]),
                                                                             threshold=[4.],
                                                                             signed=True)
        self.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def test_illegal_num_of_cluster_centers_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25, 12, 45, 11])
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=2,
                                                                             cluster_centers=cluster_centers,
                                                                             threshold=[4.],
                                                                             signed=True)
        self.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                         f'{len(cluster_centers)}', str(e.exception))

    def test_illegal_cluster_centers_range_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=2,
                                                                             cluster_centers=np.asarray([50]),
                                                                             threshold=[4.],
                                                                             signed=True,
                                                                             multiplier_n_bits=3)
        self.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def test_illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=10,
                                                                             cluster_centers=np.asarray([25]),
                                                                             threshold=[4.],
                                                                             signed=True,
                                                                             multiplier_n_bits=8)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
                         , str(e.exception))

    def test_warning_num_bit_equal_multiplier_n_bits_inferable_quantizer(self):
        with warnings.catch_warnings(record=True) as w:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                             cluster_centers=np.asarray([25]),
                                                                             threshold=[4.],
                                                                             signed=True,
                                                                             multiplier_n_bits=8)
        self.assertTrue('Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
                        'in that case, consider using SymmetricInferableQuantizer instead'
                        in [str(warning.message) for warning in w])

    def test_illegal_num_of_thresholds_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                             cluster_centers=np.asarray([25]),
                                                                             threshold=[4., 2.],
                                                                             signed=True,
                                                                             multiplier_n_bits=8)
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=3,
                                                                             cluster_centers=np.asarray([25]),
                                                                             threshold=np.asarray([4.]),
                                                                             signed=True,
                                                                             multiplier_n_bits=8)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_illegal_signed_cluster_centers_inferable_quantizer(self):
        with self.assertRaises(Exception) as e:
            qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=8,
                                                                             cluster_centers=np.asarray([-25., 85.]),
                                                                             threshold=[2.],
                                                                             signed=False)
        self.assertEqual('Expected unsigned cluster centers in unsigned activation quantization ', str(e.exception))

    def test_lut_pot_signed_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25])
        thresholds = [4.]
        num_bits = 3
        signed = True
        multiplier_n_bits = 8
        eps = 1e-8

        quantizer = qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                                                     cluster_centers=cluster_centers,
                                                                                     signed=signed,
                                                                                     threshold=thresholds,
                                                                                     multiplier_n_bits=
                                                                                     multiplier_n_bits,
                                                                                     eps=eps)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)
        self.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.assertTrue(quantizer_config['eps'] == eps)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        quantized_tensor = quantizer(input_tensor)

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(thresholds))
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

        quant_tensor_values = (cluster_centers / (2 ** (multiplier_n_bits - int(signed)))) * thresholds

        self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        tensor = tf.clip_by_value((input_tensor / (thresholds + eps)) * (2 ** (num_bits - 1)),
                                  clip_value_max=clip_max, clip_value_min=clip_min)
        tensor = tf.expand_dims(tensor, -1)
        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
        centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

        self.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * thresholds == quantized_tensor),
                        "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_lut_pot_unsigned_inferable_quantizer(self):
        cluster_centers = np.asarray([25, 85])
        thresholds = [2.]
        num_bits = 3
        signed = False
        multiplier_n_bits = 8
        eps = 1e-8

        quantizer = qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                                                     cluster_centers=cluster_centers,
                                                                                     signed=signed,
                                                                                     threshold=thresholds,
                                                                                     multiplier_n_bits=
                                                                                     multiplier_n_bits,
                                                                                     eps=eps)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)
        self.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.assertTrue(quantizer_config['eps'] == eps)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        quantized_tensor = quantizer(input_tensor)

        # Using a unsigned quantization, so we expect all values to be between 0
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** multiplier_n_bits)

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** multiplier_n_bits - 1
        clip_min = 0

        quant_tensor_values = (cluster_centers / (2 ** multiplier_n_bits)) * thresholds

        self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        tensor = tf.clip_by_value((input_tensor / (thresholds + eps)) * (2 ** multiplier_n_bits),
                                  clip_value_max=clip_max, clip_value_min=clip_min)
        tensor = tf.expand_dims(tensor, -1)

        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
        centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

        self.assertTrue(np.all(centers / (2 ** multiplier_n_bits) * thresholds == quantized_tensor),
                        "Quantized tensor values weren't assigned correctly")

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')
