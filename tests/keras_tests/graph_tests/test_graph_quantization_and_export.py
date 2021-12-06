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
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from model_compression_toolkit.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.common.bias_correction.compute_bias_correction_of_graph import \
    compute_bias_correction_of_graph
from model_compression_toolkit.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.keras.reader.reader import model_reader
from model_compression_toolkit.common.model_collector import ModelCollector
from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.keras.tensor_marking import get_node_stats_collector
from model_compression_toolkit.keras.back2framework.model_builder import model_builder
from model_compression_toolkit.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG
import tensorflow as tf



class TestGraphQuantization(unittest.TestCase):

    def test_bn_folding_mbv1(self):
        model = MobileNetV2()
        graph = model_reader(model)  # model pharsing
        fw_impl = KerasImplementation()
        tg = substitute(graph, fw_impl.get_substitutions_pre_statistics_collection())
        analyzer_graph(fw_impl.attach_sc_to_node, tg, DEFAULT_KERAS_INFO)  # mark tensors and quantization point
        mi = ModelCollector(tg, fw_impl, DEFAULT_KERAS_INFO)

        for i in range(10):
            mi.infer([np.random.randn(1, 224, 224, 3)])

        tg = set_quantization_configuration_to_graph(tg,
                                                     DEFAULTCONFIG,
                                                     DEFAULT_KERAS_INFO)
        calculate_quantization_params(tg,
                                      DEFAULT_KERAS_INFO,
                                      fw_impl=fw_impl)

        tg = compute_bias_correction_of_graph(tg, fw_info=DEFAULT_KERAS_INFO, fw_impl=fw_impl)

        tg = set_bit_widths(DEFAULTCONFIG,
                            tg,
                            DEFAULT_KERAS_INFO)

        quantize_graph_weights(tg,
                               DEFAULT_KERAS_INFO,
                               fw_impl=fw_impl)

        model, _ = model_builder(tg)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        quantized_tflite_model = converter.convert()


if __name__ == '__main__':
    unittest.main()
