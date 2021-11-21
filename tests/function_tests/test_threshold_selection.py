# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.common.quantization.quantization_params_generation import no_clipping_selection_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation import mse_selection_histogram
from model_compression_toolkit.common.collectors.histogram_collector import HistogramCollector


class TestThresholdSelection(unittest.TestCase):
    def test_no_clipping_function(self):
        x = np.random.randn(10, 10, 10)
        dummy = 0
        ml = no_clipping_selection_tensor(x, dummy, dummy)[THRESHOLD]
        self.assertTrue(ml > np.max(np.abs(x)))

    def test_mse_from_histogram(self):
        hc = HistogramCollector()
        for i in range(10):
            x = np.random.randn(10, 10, 10)
            hc.update(x)
        b, c = hc.get_histogram()
        dummy = 0
        mse_selection_histogram(b, c, dummy, 8, dummy, dummy)


if __name__ == '__main__':
    unittest.main()
