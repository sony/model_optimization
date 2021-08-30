# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import unittest
import numpy as np
from sony_model_optimization_package.common.collectors.histogram_collector import HistogramCollector, interpolate_histogram
from tests.feature_networks_tests.test_networks_runner import set_seed


class TestHistogramCollector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        set_seed()

    def test_same(self):
        hc = HistogramCollector()
        x = np.random.rand(1, 2, 3, 4)
        for i in range(100):
            hc.update(x)

        self.assertTrue(np.isclose(np.max(x), hc.max(), atol=0.001))
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
