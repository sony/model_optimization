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
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from sony_model_optimization_package.common.quantization.set_node_quantization_config import set_qcs_to_graph_nodes
from sony_model_optimization_package.keras.reader.reader import model_reader
from sony_model_optimization_package.keras.back2framework.model_collector import ModelCollector
from sony_model_optimization_package.common.quantization.quantization_analyzer import analyzer_graph
from sony_model_optimization_package.keras.tensor_marking import get_node_stats_collector
from sony_model_optimization_package.keras.back2framework.model_builder import model_builder
from sony_model_optimization_package.common.quantization.quantize_model import quantize_model, calculate_quantization_params
from sony_model_optimization_package.keras.default_framework_info import DEFAULT_KERAS_INFO
from sony_model_optimization_package.common.quantization.quantization_config import DEFAULTCONFIG
import tensorflow as tf
from sony_model_optimization_package.keras.graph_substitutions.substituter import pre_statistics_collection_substitute


class TestGraphQuantization(unittest.TestCase):

    def test_bn_folding_mbv1(self):
        model = MobileNetV2()
        graph = model_reader(model)  # model pharsing
        tg = pre_statistics_collection_substitute(graph)  # substition
        analyzer_graph(get_node_stats_collector, tg, DEFAULT_KERAS_INFO)  # mark tensors and quantization point
        mi = ModelCollector(tg)

        for i in range(10):
            mi.infer([np.random.randn(1, 224, 224, 3)])

        tg = set_qcs_to_graph_nodes(tg,
                                    DEFAULTCONFIG,
                                    DEFAULT_KERAS_INFO)
        calculate_quantization_params(tg,
                                      DEFAULT_KERAS_INFO)
        quantize_model(tg,
                       DEFAULT_KERAS_INFO)

        model, _ = model_builder(tg)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        quantized_tflite_model = converter.convert()


if __name__ == '__main__':
    unittest.main()
