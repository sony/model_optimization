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
import tensorflow as tf

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers import \
    ActivationSymmetricInferableQuantizer, ActivationPOTInferableQuantizer, ActivationUniformInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest


class TestKerasActivationsSymmetricInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        num_bits = 3
        thresholds = [4.]
        signed = True

        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=signed)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.unit_test.assertTrue(quantizer_config['threshold'] == thresholds)
        self.unit_test.assertTrue(quantizer_config['signed'] == signed)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a signed quantization, so we expect all values to be between -4 and 4
        self.unit_test.assertTrue(np.max(
            quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than maximal '
                                               f'threshold ')
        self.unit_test.assertTrue(np.min(
            quantized_tensor) >= -thresholds[0], f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.unit_test.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds[0] / (2 ** (num_bits - int(signed)))
        manually_quantized_tensor = tf.clip_by_value(np.round(input_tensor / scale), clip_value_min=-thresholds[0],
                                                     clip_value_max=thresholds[0] - scale)
        self.unit_test.assertTrue(np.all(manually_quantized_tensor.numpy() == quantized_tensor.numpy()))


class TestKerasActivationsUnsignedSymmetricInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        thresholds = [4.]
        num_bits = 2
        signed = False

        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=signed)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.unit_test.assertTrue(quantizer_config['threshold'] == thresholds)
        self.unit_test.assertTrue(quantizer_config['signed'] == signed)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        # The maximal threshold is 4 using a unsigned quantization, so we expect all values to be between 0 and 4
        self.unit_test.assertTrue(np.max(
            quantized_tensor) < thresholds[0], f'Quantized values should not contain values greater than maximal '
                                               f'threshold')
        self.unit_test.assertTrue(np.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.unit_test.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Assert all values are non-negative (unsigned quantization)
        self.unit_test.assertTrue(np.all(quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds[0] / (2 ** num_bits - int(signed))
        manually_quantized_tensor = tf.clip_by_value(np.round(input_tensor / scale), clip_value_min=0,
                                                     clip_value_max=thresholds[0] - scale)
        self.unit_test.assertTrue(np.all(manually_quantized_tensor.numpy() == quantized_tensor.numpy()))


class TestKerasActivationsIllegalPOTInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            ActivationPOTInferableQuantizer(num_bits=8,
                                            threshold=[3.],
                                            signed=True)
        self.unit_test.assertEqual('Expected threshold to be power of 2 but is [3.]', str(e.exception))


class TestKerasActivationsPOTInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        thresholds = [1.]
        num_bits = 2
        signed = True

        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    signed=signed,
                                                    threshold=thresholds)

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.unit_test.assertTrue(quantizer_config['threshold'] == thresholds)
        self.unit_test.assertTrue(quantizer_config['signed'] == signed)

        delta = thresholds - quantizer.max_range
        is_pot_delta = np.all(
            np.log2(delta) == np.log2(delta).astype(int))
        self.unit_test.assertTrue(is_pot_delta, f'Expected delta to be POT but: {delta}')

        self.unit_test.assertTrue(np.all(quantizer.min_range == -1 * thresholds))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        self.unit_test.assertTrue(np.max(fake_quantized_tensor) < thresholds[
            0], f'Quantized values should not contain values greater than threshold')
        self.unit_test.assertTrue(np.min(fake_quantized_tensor) >= -thresholds[
            0], f'Quantized values should not contain values lower than threshold')

        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(np.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds / (2 ** (num_bits - int(signed)))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=-thresholds,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))


class TestKerasActivationsUnsignedPOTInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        thresholds = [1.]
        num_bits = 2
        signed = False

        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    signed=signed,
                                                    threshold=thresholds)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.unit_test.assertTrue(quantizer_config['threshold'] == thresholds)
        self.unit_test.assertTrue(quantizer_config['signed'] == signed)

        delta = thresholds - quantizer.max_range
        is_pot_delta = np.all(
            np.log2(delta) == np.log2(delta).astype(int))
        self.unit_test.assertTrue(is_pot_delta, f'Expected delta to be POT but: {delta}')

        self.unit_test.assertTrue(np.all(quantizer.min_range == 0))

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        self.unit_test.assertTrue(np.max(fake_quantized_tensor) < thresholds[
            0], f'Quantized values should not contain values greater than threshold')

        self.unit_test.assertTrue(np.min(fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        # Assert all values are non-negative (unsigned quantization)
        self.unit_test.assertTrue(np.any(fake_quantized_tensor >= 0),
                        f'Expected all values to be non-negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = thresholds / (2 ** num_bits - int(signed))
        manually_quantized_tensor = np.round(
            tf.clip_by_value(input_tensor, clip_value_min=0,
                             clip_value_max=thresholds - scale) / scale) * scale
        self.unit_test.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))


class TestKerasActivationsUniformInferableQuantizer(BaseInferableQuantizerTest):

    def run_test(self):
        min_range = [-10.]
        max_range = [5.]
        num_bits = 2
        quantizer = ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                        min_range=min_range,
                                                        max_range=max_range)

        min_range = np.asarray(min_range)
        max_range = np.asarray(max_range)

        # check config
        quantizer_config = quantizer.get_config()
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.unit_test.assertTrue(quantizer_config['min_range'] == min_range)
        self.unit_test.assertTrue(quantizer_config['max_range'] == max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 4, 50) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        # We expect tensor values values to be between min_range to max_range
        self.unit_test.assertTrue(np.max(fake_quantized_tensor) <= max_range[
            0], f'Quantized values should not contain values greater than threshold')
        self.unit_test.assertTrue(np.min(fake_quantized_tensor) >= min_range[
            0], f'Quantized values should not contain values lower than threshold')
        self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        # Assert some values are negative
        self.unit_test.assertTrue(np.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

        # Assert manually quantized values are the same:
        scale = (max_range - min_range) / (2 ** num_bits - 1)
        manually_quantized_tensor = \
            np.round((tf.clip_by_value(input_tensor, clip_value_min=min_range,
                                       clip_value_max=max_range) - min_range) / scale) * scale + min_range
        self.unit_test.assertTrue(np.all(manually_quantized_tensor == fake_quantized_tensor.numpy()))


class TestKerasActivationsUniformInferableZeroNotInRange(BaseInferableQuantizerTest):

    def run_test(self):
        min_range = [3.]
        max_range = [10.]
        num_bits = 2

        quantizer = ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                        min_range=min_range,
                                                        max_range=max_range)

        # check config
        quantizer_config = quantizer.get_config()
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        # TODO: fix check
        # self.assertTrue(quantizer_config['min_range'] == min_range)
        # self.assertTrue(quantizer_config['max_range'] == max_range)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 4, 50) * 100 - 50, tf.float32)
        fake_quantized_tensor = quantizer(input_tensor)

        # We expect each channel values to be between min_range to max_range for each channel
        for i in range(len(min_range)):
            channel_slice_i = fake_quantized_tensor[:, :, i, :]
            self.unit_test.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(channel_slice_i))} unique values')
            self.unit_test.assertTrue(0 in np.unique(channel_slice_i),
                            f'zero should be in quantization range, but quantized values are in set: '
                            f'{np.unique(channel_slice_i)}')

