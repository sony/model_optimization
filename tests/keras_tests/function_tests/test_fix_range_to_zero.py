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

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import \
    uniform_quantize_tensor, get_output_shape


class TestFixRangeToZero(unittest.TestCase):

    def test_uniform_quantize_fix_range_to_zero(self):
        self.fix_range_test_run(min_max_range=[-1.5, 1.3333], data_range=[-1.5, 1.3333], n_bits=8)
        self.fix_range_test_run(min_max_range=[-1.5, 1.3333], data_range=[-1.5, 1.3333], n_bits=6)
        self.fix_range_test_run(min_max_range=[-1.5, 1.3333], data_range=[-1.5, 1.3333], n_bits=4)

        self.fix_range_test_run(min_max_range=[-1.5, -1.3333], data_range=[-1.5, 1.3333], n_bits=8)
        self.fix_range_test_run(min_max_range=[-1.5, -1.3333], data_range=[-1.5, 1.3333], n_bits=6)
        self.fix_range_test_run(min_max_range=[-1.5, -1.3333], data_range=[-1.5, 1.3333], n_bits=4)

        self.fix_range_test_run(min_max_range=[1.4, 2.3333], data_range=[-1.5, 2.3333], n_bits=8)
        self.fix_range_test_run(min_max_range=[1.4, 2.3333], data_range=[-1.5, 2.3333], n_bits=6)
        self.fix_range_test_run(min_max_range=[1.4, 2.3333], data_range=[-1.5, 2.3333], n_bits=4)

        self.fix_range_per_channel_test_run(min_range_bounds=[-1.5, -0.3333], max_range_bounds=[1, 2.3333],
                                            data_range=[-2, 2], n_bits=8, shape=[2, 10, 10, -1], channel_axis=3)

        self.fix_range_per_channel_test_run(min_range_bounds=[-1.5, 1.4], max_range_bounds=[2.3333, 4.5],
                                            data_range=[-2, -0.222], n_bits=8, shape=[2, 10, -1, 10], channel_axis=2)

        self.fix_range_per_channel_test_run(min_range_bounds=[-1.5, -0.3333], max_range_bounds=[1, 2.3333],
                                            data_range=[-2, 2], n_bits=6, shape=[2, 10, 10, -1], channel_axis=3)

        self.fix_range_per_channel_test_run(min_range_bounds=[-1.5, 1.4], max_range_bounds=[2.3333, 4.5],
                                            data_range=[-2, -0.222], n_bits=4, shape=[2, 10, -1, 10], channel_axis=2)

    def fix_range_test_run(self, min_max_range, data_range, n_bits):
        a, b = data_range
        tensor_data = np.linspace(start=a, stop=b, num=2000)
        q = uniform_quantize_tensor(tensor_data, range_min=min_max_range[0], range_max=min_max_range[1], n_bits=n_bits)
        self.assertTrue((q == 0).any())

    def fix_range_per_channel_test_run(self, min_range_bounds, max_range_bounds, data_range, n_bits, shape, channel_axis):
        a, b = data_range
        tensor_data = np.linspace(start=a, stop=b, num=2000)
        tensor_data = np.reshape(tensor_data, newshape=shape)
        min_range = np.reshape(np.linspace(start=min_range_bounds[0], stop=min_range_bounds[1],
                                           num=tensor_data.shape[channel_axis]), get_output_shape(shape, channel_axis))
        max_range = np.reshape(np.linspace(start=max_range_bounds[0], stop=max_range_bounds[1],
                                           num=tensor_data.shape[channel_axis]), get_output_shape(shape, channel_axis))
        q = uniform_quantize_tensor(tensor_data, range_min=min_range, range_max=max_range, n_bits=n_bits)
        self.assertTrue((q == 0).any())


if __name__ == '__main__':
    unittest.main()
