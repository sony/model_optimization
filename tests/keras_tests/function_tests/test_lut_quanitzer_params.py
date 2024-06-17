# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

import math
import random
import unittest
import numpy as np

from model_compression_toolkit.constants import LUT_VALUES, SCALE_PER_CHANNEL
from model_compression_toolkit.core.common.quantization.quantization_params_generation.lut_kmeans_params import \
    lut_kmeans_tensor


class TestLUTQuantizerParams(unittest.TestCase):

    def test_properties(self):
        channel_axis = random.choice([0, 1, 2, 3])
        tensor_data = np.random.randn(3, 4, 5, 6)
        quantization_params = lut_kmeans_tensor(tensor_data=tensor_data,
                                                p=2,
                                                n_bits=4,
                                                per_channel=True,
                                                channel_axis=channel_axis)[0]
        lut_values = quantization_params[LUT_VALUES]
        scales_per_channel = quantization_params[SCALE_PER_CHANNEL]
        # check size of scales
        self.assertTrue(scales_per_channel.shape[channel_axis] == tensor_data.shape[channel_axis])
        self.assertTrue(len(scales_per_channel.shape) == len(tensor_data.shape))
        # check that all scales are power of 2
        self.assertTrue(np.all([math.log2(n).is_integer() for n in list(scales_per_channel.flatten())]))
        self.assertTrue(len(np.unique(lut_values.flatten())) <= len(np.unique(tensor_data.flatten())))

    def test_properties_with_fewer_data(self):
        # check when the number of values of the tensor_data is lower than 2**n_bits
        channel_axis = random.choice([0, 1, 2])
        tensor_data = np.random.randn(3, 2, 2)
        quantization_params = lut_kmeans_tensor(tensor_data=tensor_data,
                                                p=2,
                                                n_bits=4,
                                                per_channel=True,
                                                channel_axis=channel_axis)[0]
        lut_values = quantization_params[LUT_VALUES]
        scales_per_channel = quantization_params[SCALE_PER_CHANNEL]
        # check size of scales
        self.assertTrue(scales_per_channel.shape[channel_axis] == tensor_data.shape[channel_axis])
        self.assertTrue(len(scales_per_channel.shape) == len(tensor_data.shape))
        # check that all scales are power of 2
        self.assertTrue(np.all([math.log2(n).is_integer() for n in list(scales_per_channel.flatten())]))
        # len(unique(tensor_data)) < 2 ** n_bits then lut_values = len(unique(tensor_data))
        is_zero_added = 0 in lut_values and 0 not in tensor_data  # check if 0 was added to LUT values in quantization.
        self.assertTrue(len(lut_values.flatten()) - int(is_zero_added) <= len(np.unique(tensor_data.flatten())))


if __name__ == '__main__':
    unittest.main()
