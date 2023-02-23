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
import torch
import warnings

from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizer_utils import \
    get_working_device
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers.weights_inferable_quantizers.weights_lut_pot_inferable_quantizer import \
    WeightsLUTPOTInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest


class BasePytorchWeightsLUTQuantizerTest(BaseInferableQuantizerTest):

    def illegal_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                         per_channel, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.unit_test.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def illegal_num_of_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                                per_channel, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=2,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.unit_test.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                                   f'{len(cluster_centers)}', str(e.exception))

    def illegal_cluster_centers_range_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                               per_channel, channel_axis,
                                                               multiplier_n_bits):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits)
        self.unit_test.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                               cluster_centers,
                                                                               num_bits,
                                                                               per_channel, channel_axis,
                                                                               multiplier_n_bits):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits)
        self.unit_test.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
                                   , str(e.exception))

    def warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                         cluster_centers,
                                                                         num_bits,
                                                                         per_channel, channel_axis,
                                                                         multiplier_n_bits):
        with warnings.catch_warnings(record=True) as w:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits)
        self.unit_test.assertTrue(
            'Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
            'in that case, consider using SymmetricInferableQuantizer instead'
            in [str(warning.message) for warning in w])

    def illegal_num_of_thresholds_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                           per_channel, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.unit_test.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2',
                                   str(e.exception))

    def illegal_threshold_type_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                        per_channel, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.unit_test.assertEqual('Threshold is expected to be numpy array, but is of type <class \'list\'>',
                                   str(e.exception))

    def illegal_threshold_shape_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                         per_channel, channel_axis):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.unit_test.assertEqual(f'Threshold is expected to be flatten, but of shape {threshold.shape}',
                                   str(e.exception))

    def missing_channel_axis_inferable_quantizer(self, inferable_quantizer, threshold, cluster_centers,
                                                 per_channel):
        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold)
        self.unit_test.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def weights_inferable_quantizer_test(self, inferable_quantizer, num_bits, threshold, cluster_centers,
                                         per_channel, channel_axis, multiplier_n_bits):
        quantizer = inferable_quantizer(num_bits=num_bits,
                                        per_channel=per_channel,
                                        cluster_centers=cluster_centers,
                                        threshold=threshold,
                                        channel_axis=channel_axis,
                                        multiplier_n_bits=multiplier_n_bits)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))
        max_threshold = np.max(np.abs(threshold))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - 1))

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

        clip_max = 2 ** (multiplier_n_bits - 1) - 1
        clip_min = -2 ** (multiplier_n_bits - 1)

        if per_channel:
            for i in range(len(threshold)):
                channel_slice_i = fake_quantized_tensor[:, :, :, i]
                channel_quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold[i]
                self.unit_test.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                                          f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                          f'{len(np.unique(channel_slice_i))} unique values')
                self.unit_test.assertTrue(np.all(np.unique(channel_slice_i) == np.sort(channel_quant_tensor_values)))

                # Check quantized tensor assigned correctly
                tensor = torch.clip((input_tensor / threshold[i]) * (2 ** (multiplier_n_bits - 1)),
                                    min=clip_min, max=clip_max)
                tensor = tensor[:, :, :, i].unsqueeze(-1)
                expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
                cluster_assignments = torch.argmin(torch.abs(tensor - expanded_cluster_centers), dim=-1)
                centers = cluster_centers.flatten()[cluster_assignments]

                self.unit_test.assertTrue(
                    np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[i] == channel_slice_i),
                    "Quantized tensor values weren't assigned correctly")

        else:
            quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold
            self.unit_test.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                      f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                      f'{len(np.unique(fake_quantized_tensor))} unique values')
            self.unit_test.assertTrue(np.all(np.unique(fake_quantized_tensor)
                                             == np.sort(quant_tensor_values)))

            # Check quantized tensor assigned correctly
            tensor = torch.clip((input_tensor / threshold) * (2 ** (multiplier_n_bits - 1)),
                                min=clip_min, max=clip_max)
            tensor = tensor.unsqueeze(-1)
            expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
            cluster_assignments = torch.argmin(torch.abs(tensor - expanded_cluster_centers), dim=-1)
            centers = cluster_centers.flatten()[cluster_assignments]

            self.unit_test.assertTrue(
                np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold == fake_quantized_tensor),
                "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.unit_test.assertTrue(np.any(fake_quantized_tensor < 0),
                                  f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')


class TestPytorchWeightsSymmetricLUTQuantizerAssertions(BasePytorchWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25.6, 25])
        per_channel = False
        channel_axis = None

        threshold = np.asarray([2.])
        self.illegal_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                              threshold=threshold,
                                                              cluster_centers=cluster_centers, per_channel=per_channel,
                                                              channel_axis=channel_axis)

        cluster_centers = np.asarray([-25, 25, 3, 19, 55])
        self.illegal_num_of_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                     threshold=threshold,
                                                                     cluster_centers=cluster_centers,
                                                                     per_channel=per_channel,
                                                                     channel_axis=channel_axis)
        multiplier_n_bits = 5
        cluster_centers = np.asarray([-25, 25])
        self.illegal_cluster_centers_range_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                    threshold=threshold,
                                                                    cluster_centers=cluster_centers,
                                                                    per_channel=per_channel,
                                                                    channel_axis=channel_axis,
                                                                    multiplier_n_bits=multiplier_n_bits)

        multiplier_n_bits = 8
        num_bits = 10
        self.illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits)

        multiplier_n_bits = 8
        num_bits = 8
        self.warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits)

        threshold = [3., 2.]
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                             threshold=threshold,
                                                             cluster_centers=cluster_centers, per_channel=per_channel,
                                                             channel_axis=channel_axis)

        threshold = np.array([[3., 2.], [2., 5.]])
        self.illegal_threshold_shape_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                              threshold=threshold,
                                                              cluster_centers=cluster_centers, per_channel=per_channel,
                                                              channel_axis=channel_axis)

        threshold = np.asarray([2., 7.])
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                threshold=threshold, cluster_centers=cluster_centers,
                                                                per_channel=per_channel, channel_axis=channel_axis)

        threshold = np.asarray([2.])
        per_channel = True
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=inferable_quantizer,
                                                      threshold=threshold,
                                                      cluster_centers=cluster_centers, per_channel=per_channel)


class TestPytorchWeightsSymmetricLUTQuantizer(BasePytorchWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25, 25])
        per_channel = True
        num_bits = 3

        # test per channel
        threshold = np.asarray([3., 8., 7.])
        channel_axis = 3
        multiplier_n_bits = 8
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              multiplier_n_bits=multiplier_n_bits)

        # test per channel and channel axis is not last
        threshold = np.asarray([3., 8., 7.])
        channel_axis = 1
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              multiplier_n_bits=multiplier_n_bits)

        # test per tensor
        threshold = np.asarray([3.])
        channel_axis = None
        per_channel = False
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              multiplier_n_bits=multiplier_n_bits)


class TestPytorchWeightsLUTPOTQuantizerAssertions(BasePytorchWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTPOTInferableQuantizer

        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                cluster_centers=np.asarray([25., 85.]),
                                per_channel=False,
                                # Not POT threshold
                                threshold=np.asarray([3]))
        self.unit_test.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                cluster_centers=np.asarray([25., 85.]),
                                per_channel=False,
                                # More than one threshold in per-tensor
                                # quantization
                                threshold=np.asarray([2, 3]))
        self.unit_test.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2', str(e.exception))

        cluster_centers = np.asarray([-25.6, 25])
        per_channel = False
        channel_axis = None

        threshold = np.asarray([2.])
        self.illegal_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                              threshold=threshold,
                                                              cluster_centers=cluster_centers, per_channel=per_channel,
                                                              channel_axis=channel_axis)

        cluster_centers = np.asarray([-25, 25, 3, 19, 55])
        self.illegal_num_of_cluster_centers_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                     threshold=threshold,
                                                                     cluster_centers=cluster_centers,
                                                                     per_channel=per_channel,
                                                                     channel_axis=channel_axis)

        multiplier_n_bits = 5
        cluster_centers = np.asarray([-25, 25])
        self.illegal_cluster_centers_range_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                    threshold=threshold,
                                                                    cluster_centers=cluster_centers,
                                                                    per_channel=per_channel,
                                                                    channel_axis=channel_axis,
                                                                    multiplier_n_bits=multiplier_n_bits)
        multiplier_n_bits = 8
        num_bits = 10
        self.illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits)

        num_bits = 8
        self.warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=inferable_quantizer,
            threshold=threshold, cluster_centers=cluster_centers,
            per_channel=per_channel, channel_axis=channel_axis,
            multiplier_n_bits=multiplier_n_bits, num_bits=num_bits)

        num_bits = 3
        threshold = [4., 2.]
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                             threshold=threshold,
                                                             cluster_centers=cluster_centers, per_channel=per_channel,
                                                             channel_axis=channel_axis)

        threshold = np.array([[4., 2.], [2., 8.]])
        self.illegal_threshold_shape_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                              threshold=threshold,
                                                              cluster_centers=cluster_centers, per_channel=per_channel,
                                                              channel_axis=channel_axis)

        threshold = np.asarray([2., 8.])
        multiplier_n_bits = 7
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=inferable_quantizer,
                                                                threshold=threshold, cluster_centers=cluster_centers,
                                                                per_channel=per_channel, channel_axis=channel_axis)

        threshold = np.asarray([2.])
        per_channel = True
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=inferable_quantizer,
                                                      threshold=threshold,
                                                      cluster_centers=cluster_centers, per_channel=per_channel)


class TestPytorchWeightsPOTLUTQuantizer(BasePytorchWeightsLUTQuantizerTest):

    def run_test(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25, 25])
        per_channel = True
        num_bits = 3
        multiplier_n_bits = 7

        # test per channel
        threshold = np.asarray([2., 8., 16.])
        channel_axis = 3
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              multiplier_n_bits=multiplier_n_bits)

        # test per channel and channel axis is not last
        threshold = np.asarray([2., 8., 16.])
        channel_axis = 1
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              multiplier_n_bits=multiplier_n_bits)

        # test per tensor
        threshold = np.asarray([4.])
        channel_axis = None
        per_channel = False
        self.weights_inferable_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                              threshold=threshold, cluster_centers=cluster_centers,
                                              per_channel=per_channel, channel_axis=channel_axis,
                                              multiplier_n_bits=multiplier_n_bits)
