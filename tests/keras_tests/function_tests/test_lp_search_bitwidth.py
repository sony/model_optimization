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
import copy

import numpy as np
import unittest
from functools import partial
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from model_compression_toolkit.common.constants import TENSORFLOW
from model_compression_toolkit.common.mixed_precision.distance_weighting import get_average_weights
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG, MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.mixed_precision.mixed_precision_search_facade import search_bit_width, \
    BitWidthSearchMethod
from model_compression_toolkit.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.common.model_collector import ModelCollector

from model_compression_toolkit import get_model, DEFAULTCONFIG
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from model_compression_toolkit.keras.constants import DEFAULT_HWM
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from tests.common_tests.helpers.generate_test_hw_model import generate_test_hw_model


class TestLpSearchBitwidth(unittest.TestCase):

    def test_search(self):
        target_kpi = KPI(2)
        layer_to_kpi_mapping = {0: {2: KPI(1),
                                    1: KPI(2),
                                    0: KPI(3)}}

        bit_cfg = mp_integer_programming_search({0: [0, 1, 2]},
                                                lambda x, y: 0,
                                                lambda x: layer_to_kpi_mapping[0][x[0]],
                                                target_kpi)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_kpi = KPI(0)  # Infeasible solution!
        with self.assertRaises(Exception):
            bit_cfg = mp_integer_programming_search({0: [0, 1, 2]},
                                                    lambda x, y: 0,
                                                    lambda x: layer_to_kpi_mapping[0][x[0]],
                                                    target_kpi)

        bit_cfg = mp_integer_programming_search({0: [0, 1, 2]},
                                                lambda x, y: 0,
                                                lambda x: layer_to_kpi_mapping[0][x[0]],
                                                KPI(np.inf))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 2)


class TestSearchBitwidthConfiguration(unittest.TestCase):

    def test_search_engine(self):

        qc = MixedPrecisionQuantizationConfig(DEFAULTCONFIG,
                                              compute_mse,
                                              get_average_weights,
                                              num_of_images=1)
        KERAS_DEFAULT_MODEL = get_model(TENSORFLOW, DEFAULT_HWM)
        fw_info = DEFAULT_KERAS_INFO
        in_model = MobileNetV2()
        keras_impl = KerasImplementation()

        def dummy_representative_dataset():
            return None

        graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading
        graph.set_fw_info(fw_info)
        graph.set_fw_hw_model(fw_hw_model)
        graph = set_quantization_configuration_to_graph(graph=graph,
                                                        quant_config=qc)

        for node in graph.nodes:
            node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                             fw_info=fw_info,
                                                             graph=graph)

        analyzer_graph(keras_impl.attach_sc_to_node,
                       graph,
                       fw_info)

        mi = ModelCollector(graph,
                            fw_info=DEFAULT_KERAS_INFO,
                            fw_impl=keras_impl)

        for i in range(10):
            mi.infer([np.random.randn(1, 224, 224, 3)])

        calculate_quantization_params(graph,
                                      fw_info,
                                      fw_impl=keras_impl)
        keras_sens_eval = partial(keras_impl.get_sensitivity_evaluation_fn,
                                  representative_data_gen=lambda: [np.random.random((1, 224, 224, 3))],
                                  fw_info=fw_info)

        cfg = search_bit_width(graph,
                               qc,
                               DEFAULT_KERAS_INFO,
                               KPI(np.inf),
                               keras_sens_eval,
                               BitWidthSearchMethod.INTEGER_PROGRAMMING)

        with self.assertRaises(Exception):
            cfg = search_bit_width(graph,
                                   qc,
                                   DEFAULT_KERAS_INFO,
                                   KPI(np.inf),
                                   keras_sens_eval,
                                   None)

        with self.assertRaises(Exception):
            cfg = search_bit_width(graph,
                                   qc,
                                   DEFAULT_KERAS_INFO,
                                   None,
                                   keras_sens_eval,
                                   BitWidthSearchMethod.INTEGER_PROGRAMMING)


if __name__ == '__main__':
    unittest.main()
