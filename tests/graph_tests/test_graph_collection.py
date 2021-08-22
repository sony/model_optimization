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
from tensorflow.keras.applications.mobilenet import MobileNet
from network_optimization_package.keras.reader.reader import model_reader
from network_optimization_package.keras.back2framework.model_collector import ModelCollector
from network_optimization_package.common.quantization.quantization_analyzer import analyzer_graph
from network_optimization_package.keras.tensor_marking import get_node_stats_collector
from network_optimization_package.keras.default_framework_info import DEFAULT_KERAS_INFO
from network_optimization_package.keras.graph_substitutions.substituter import pre_statistics_collection_substitute


class TestGraphStatisticCollection(unittest.TestCase):

    def test_bn_folding_mbv1(self):
        model = MobileNet()
        graph = model_reader(model)  # model pharsing
        tg = pre_statistics_collection_substitute(graph)  # substition
        analyzer_graph(get_node_stats_collector, tg, DEFAULT_KERAS_INFO)  # Mark point for collection of statistics
        mi = ModelCollector(tg)
        for i in range(10):
            mi.infer([np.random.randn(1, 224, 224, 3)])


if __name__ == '__main__':
    unittest.main()
