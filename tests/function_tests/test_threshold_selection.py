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
from network_optimization_package.common.constants import THRESHOLD
from network_optimization_package.common.threshold_selection.no_clipping import no_clipping_selection_tensor
from network_optimization_package.common.threshold_selection.mse_selection import mse_selection_histogram
from network_optimization_package.common.collectors.histogram_collector import HistogramCollector


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
        b,c = hc.get_histogram()
        dummy = 0
        mse_selection_histogram(b, c, dummy, 8, dummy, dummy)


if __name__ == '__main__':
    unittest.main()
