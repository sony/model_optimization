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
from model_compression_toolkit.quantizers_infrastructure.common.constants import MULTIPLIER_N_BITS


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
        self.assertEqual('Expected threshold to be power of 2 but is [3.]', str(e.exception))

    def test_lut_pot_signed_inferable_quantizer(self):
        cluster_centers = np.asarray([-25, 25])
        thresholds = [4.]
        num_bits = 3
        signed = True

        quantizer = qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                                                     cluster_centers=cluster_centers,
                                                                                     signed=signed,
                                                                                     threshold=thresholds)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        quant_tensor_values = tf.concat([(center / (2 ** (MULTIPLIER_N_BITS - 1))) * thresholds for
                                         center in cluster_centers], 0)

        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        self.assertTrue(np.all(np.unique(fake_quantized_tensor) == np.sort(quant_tensor_values)))

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

    def test_lut_pot_unsigned_inferable_quantizer(self):
        cluster_centers = np.asarray([25, 75])
        thresholds = [2.]
        num_bits = 3
        signed = False
        quantizer = qi.keras_inferable_quantizers.ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                                                     cluster_centers=cluster_centers,
                                                                                     threshold=thresholds,
                                                                                     signed=signed)
        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        quant_tensor_values = (cluster_centers / (2 ** (MULTIPLIER_N_BITS - 1))) * thresholds
        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        self.assertTrue(np.all(np.unique(fake_quantized_tensor) == np.sort(quant_tensor_values)))

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.all(fake_quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {fake_quantized_tensor}')
