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
import tensorflow as tf

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.weights_inferable_quantizers.weights_lut_pot_inferable_quantizer import \
    WeightsLUTPOTInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest


class BaseKerasWeightsLUTQuantizerTest(BaseInferableQuantizerTest):

    def illegal_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                         per_channel, input_rank, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.unit_test.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def illegal_num_of_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                                per_channel, input_rank, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=2,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.unit_test.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                         f'{len(cluster_centers)}', str(e.exception))

    def illegal_cluster_centers_range_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                               per_channel, input_rank, channel_axis,
                                                               multiplier_n_bits):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits,
                                input_rank=input_rank)
        self.unit_test.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                               cluster_centers,
                                                                               num_bits, input_rank,
                                                                               per_channel, channel_axis,
                                                                               multiplier_n_bits):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits,
                                input_rank=input_rank)
        self.unit_test.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
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
        self.unit_test.assertTrue('Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
                        'in that case, consider using SymmetricInferableQuantizer instead'
                        in [str(warning.message) for warning in w])

    def illegal_num_of_thresholds_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                           per_channel, channel_axis, input_rank):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.unit_test.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2', str(e.exception))

    def illegal_threshold_type_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                        per_channel, channel_axis, input_rank):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                input_rank=input_rank)
        self.unit_test.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def missing_channel_axis_inferable_quantizer(self, inferable_quantizer, threshold, cluster_centers,
                                                 per_channel, input_rank):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                input_rank=input_rank)
        self.unit_test.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def missing_input_rank_inferable_quantizer(self, inferable_quantizer, threshold, cluster_centers,
                                               per_channel, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.unit_test.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

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
        self.unit_test.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.unit_test.assertTrue(np.all(quantizer_config['threshold'] == np.asarray(threshold)))
        self.unit_test.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.unit_test.assertTrue(quantizer_config['per_channel'] == per_channel)
        self.unit_test.assertTrue(quantizer_config['channel_axis'] == channel_axis)
        self.unit_test.assertTrue(quantizer_config['input_rank'] == input_rank)
        self.unit_test.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.unit_test.assertTrue(quantizer_config['eps'] == eps)

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

        self.unit_test.assertTrue(quantized_tensor.shape == input_tensor.shape, f'Quantized tensor should be in the same shape '
                                                                      f'as the input tensor')

        # return the output's channel axis to the last axis
        # change the input only when channel_axis is not the last axis
        quantized_tensor = tf.transpose(quantized_tensor, perm=perm_vec)

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(threshold))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - 1))

        self.unit_test.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.unit_test.assertTrue(np.min(
            quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.unit_test.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (multiplier_n_bits - 1) - 1
        clip_min = -2 ** (multiplier_n_bits - 1)

        if per_channel:
            for i in range(len(threshold)):
                channel_slice_i = quantized_tensor[:, :, :, i]
                channel_quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold[i]
                self.unit_test.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                                f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                f'{len(np.unique(channel_slice_i))} unique values')
                self.unit_test.assertTrue(np.all(np.unique(channel_slice_i) == np.sort(channel_quant_tensor_values)))

                # Check quantized tensor assigned correctly
                tensor = tf.clip_by_value((input_tensor / (threshold[i] + eps)) * (2 ** (num_bits - 1)),
                                          clip_value_max=clip_max, clip_value_min=clip_min)
                tensor = tf.expand_dims(tf.transpose(tensor, perm=perm_vec)[:, :, :, i], -1)
                expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
                cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
                centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

                self.unit_test.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[i] == channel_slice_i),
                                "Quantized tensor values weren't assigned correctly")
        else:
            quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold
            self.unit_test.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(quantized_tensor))} unique values')
            self.unit_test.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

            # Check quantized tensor assigned correctly
            tensor = tf.clip_by_value((input_tensor / (threshold[0] + eps)) * (2 ** (num_bits - 1)),
                                      clip_value_max=clip_max, clip_value_min=clip_min)
            tensor = tf.expand_dims(tensor, -1)
            expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
            cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
            centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

            self.unit_test.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[0] == quantized_tensor),
                            "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')


class TestKerasWeightsSymmetricLUTQuantizerAssertions(BaseKerasWeightsLUTQuantizerTest):

    def run_test(self):
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


class TestKerasWeightsSymmetricLUTQuantizer(BaseKerasWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25, 25])
        per_channel = True
        input_rank = 4
        num_bits = 8

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


class TestKerasWeightsLUTPOTQuantizerAssertions(BaseKerasWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTPOTInferableQuantizer

        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=False,
                                cluster_centers=np.asarray([25., 85.]),
                                threshold=[3.],
                                channel_axis=None,
                                input_rank=4)
        self.unit_test.assertEqual('Expected threshold to be power of 2 but is 3.0', str(e.exception))

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


class TestKerasWeightsPOTLUTQuantizer(BaseKerasWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer

        cluster_centers = np.asarray([-25, 25])
        input_rank = 4
        num_bits = 8

        # test per channel
        per_channel = True
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
