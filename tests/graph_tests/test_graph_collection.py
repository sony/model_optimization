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
from tensorflow.keras.applications.mobilenet import MobileNet
from model_compression_toolkit.keras.reader.reader import model_reader
from model_compression_toolkit.keras.back2framework.model_collector import ModelCollector
from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.keras.tensor_marking import get_node_stats_collector
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.graph_substitutions.substituter import pre_statistics_collection_substitute


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
