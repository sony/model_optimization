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
from model_compression_toolkit.common.collectors.histogram_collector import HistogramCollector, interpolate_histogram


class TestHistogramCollector(unittest.TestCase):

    def test_same(self):
        hc = HistogramCollector()
        x = np.random.rand(1, 2, 3, 4)
        for i in range(100):
            hc.update(x)

        self.assertTrue(np.isclose(np.max(x), hc.max(), atol=(x.max() - x.min()) / 2048))
        self.assertTrue(np.isclose(np.min(x), hc.min()))

    def test_update_hist(self):
        hc = HistogramCollector()
        x = 0.1 * np.random.rand(1, 2, 3, 4) + 0.1
        hc.update(x)
        for i in range(1000):
            x = np.random.rand(1, 2, 3, 4)
            hc.update(x)
        self.assertTrue(hc.max() > 0.9)
        self.assertTrue(hc.min() < 0.1)

    def test_same_value(self):
        hc = HistogramCollector()
        x = np.ones([100, 100])
        hc.update(x)
        self.assertTrue(hc.max() == 1.0)
        self.assertTrue(hc.min() == 1.0)

    def test_inter_histogram(self):
        x = np.random.rand(1, 2, 3, 4)
        bins = np.linspace(-2, 2, num=100)
        c, b = np.histogram(x, bins=10)
        interpolate_histogram(bins, b, c)
        self.assertTrue(True)  # Just check it works


if __name__ == '__main__':
    unittest.main()
