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


import unittest
import numpy as np
import model_compression_toolkit.core.common.quantization.quantization_config as qc
from model_compression_toolkit.core.common.constants import THRESHOLD
from model_compression_toolkit.core.common.quantization.quantization_params_generation.error_functions import _mse_error_histogram
from model_compression_toolkit.core.common.collectors.histogram_collector import HistogramCollector
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import power_of_two_selection_tensor

class TestThresholdSelection(unittest.TestCase):
    def test_no_clipping_function(self):
        x = np.random.randn(10, 10, 10)
        dummy = 0
        ml = power_of_two_selection_tensor(x, dummy, dummy, quant_error_method=qc.QuantizationErrorMethod.NOCLIPPING)[THRESHOLD]
        self.assertTrue(ml > np.max(np.abs(x)))

    def test_mse_from_histogram(self):
        hc = HistogramCollector()
        for i in range(10):
            x = np.random.randn(10, 10, 10)
            hc.update(x)
        b, c = hc.get_histogram()
        dummy = 0
        _mse_error_histogram(b, c, dummy, 8)


if __name__ == '__main__':
    unittest.main()
