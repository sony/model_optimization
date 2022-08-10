# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import calculate_delta, quantize_tensor


def ground_truth_quantize_tensor(tensor_data, threshold, n_bits, signed):
    # Compute the step size of quantized values.
    delta = calculate_delta(threshold,
                            n_bits,
                            signed)

    # Quantize the data between min/max of quantization range.
    q = delta * np.round(tensor_data / delta)
    return np.clip(q,
                   a_min=-threshold * int(signed),
                   a_max=threshold - delta)


class TestUniformQuantizeTensor(unittest.TestCase):

    def test_uniform_quantize_tensor_function(self):
        num_channels = 32
        kernel = 16
        channel_axis = 2
        random_tensor = np.random.normal(size=[kernel, kernel, num_channels, num_channels])

        # power-of-two threshold
        self.run_quantization_validation(random_tensor, threshold=2 ** 4, n_bits=8, signed=False)
        self.run_quantization_validation(random_tensor, threshold=2 ** 4, n_bits=8, signed=True)
        self.run_quantization_validation(random_tensor, threshold=2 ** 4, n_bits=6, signed=True)

        # per-channel
        output_shape = [-1 if i is channel_axis else 1 for i in range(num_channels)]

        # power-of-two thresholds per-channel
        threshold = np.reshape(np.array([2 ** 2, 2 ** 3, 2 ** 4, 2 ** 3]), output_shape)
        self.run_quantization_validation(random_tensor, threshold=threshold, n_bits=8, signed=False, per_channel=True,
                                         channel_axis=channel_axis)
        self.run_quantization_validation(random_tensor, threshold=threshold, n_bits=8, signed=True, per_channel=True,
                                         channel_axis=channel_axis)
        self.run_quantization_validation(random_tensor, threshold=threshold, n_bits=6, signed=True, per_channel=True,
                                         channel_axis=channel_axis)

    def run_quantization_validation(self, tensor_data, threshold, n_bits, signed, per_channel=False, channel_axis=1):
        gt_quantized_tensor = \
            ground_truth_quantize_tensor(tensor_data, threshold=threshold, n_bits=n_bits, signed=signed)
        quantized_tensor = \
            quantize_tensor(tensor_data, threshold=threshold, n_bits=n_bits, signed=signed)
        self.assertTrue(gt_quantized_tensor.shape == quantized_tensor.shape)
        self.assertTrue((gt_quantized_tensor == quantized_tensor).all())


if __name__ == '__main__':
    unittest.main()
