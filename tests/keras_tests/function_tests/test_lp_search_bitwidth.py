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
import numpy as np
import unittest
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_average_weights
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI, KPITarget
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width, \
    BitWidthSearchMethod
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.core.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_keras_tpc
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model


class MockReconstructionHelper:
    def __init__(self):
        pass

    def reconstruct_config_from_virtual_graph(self,
                                              max_kpi_config,
                                              changed_virtual_nodes_idx=None,
                                              original_base_config=None):
        return max_kpi_config


class MockMixedPrecisionSearchManager:
    def __init__(self, layer_to_kpi_mapping):
        self.layer_to_bitwidth_mapping = {0: [0, 1, 2]}
        self.layer_to_kpi_mapping = layer_to_kpi_mapping
        self.compute_metric_fn = lambda x, y=None, z=None: 0
        self.min_kpi = {KPITarget.WEIGHTS: [[1], [1], [1]],
                        KPITarget.ACTIVATION: [[1], [1], [1]],
                        KPITarget.TOTAL: [[2], [2], [2]],
                        KPITarget.BOPS: [[1], [1], [1]]}  # minimal kpi in the tests layer_to_kpi_mapping
        self.compute_kpi_functions = {KPITarget.WEIGHTS: (None, lambda v: [sum(v)]),
                                      KPITarget.ACTIVATION: (None, lambda v: [i for i in v]),
                                      KPITarget.TOTAL: (None, lambda v: [sum(v[0]) + i for i in v[1]]),
                                      KPITarget.BOPS: (None, lambda v: [sum(v)])}
        self.max_kpi_config = [0]
        self.config_reconstruction_helper = MockReconstructionHelper()

    def compute_kpi_matrix(self, target):
        # minus 1 is normalization by the minimal kpi (which is always 1 in this test)
        if target == KPITarget.WEIGHTS:
            kpi_matrix = [np.flip(np.array([kpi.weights_memory - 1 for _, kpi in self.layer_to_kpi_mapping[0].items()]))]
        elif target == KPITarget.ACTIVATION:
            kpi_matrix = [np.flip(np.array([kpi.activation_memory - 1 for _, kpi in self.layer_to_kpi_mapping[0].items()]))]
        elif target == KPITarget.TOTAL:
            kpi_matrix = [np.flip(np.array([kpi.total_memory - 1 for _, kpi in self.layer_to_kpi_mapping[0].items()])),
                          np.flip(np.array([kpi.total_memory - 1 for _, kpi in self.layer_to_kpi_mapping[0].items()]))]
        elif target == KPITarget.BOPS:
            kpi_matrix = [np.flip(np.array([kpi.bops - 1 for _, kpi in self.layer_to_kpi_mapping[0].items()]))]
        else:
            # not supposed to get here
            kpi_matrix = []

        return np.array(kpi_matrix)


class TestLpSearchBitwidth(unittest.TestCase):

    def test_search_weights_only(self):
        target_kpi = KPI(weights_memory=2)
        layer_to_kpi_mapping = {0: {2: KPI(weights_memory=1),
                                    1: KPI(weights_memory=2),
                                    0: KPI(weights_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_kpi_mapping)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=target_kpi)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_kpi = KPI(weights_memory=0)  # Infeasible solution!
        with self.assertRaises(Exception):
            bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                    target_kpi=target_kpi)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=KPI(weights_memory=np.inf))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 2)

    def test_search_activation_only(self):
        target_kpi = KPI(activation_memory=2)
        layer_to_kpi_mapping = {0: {2: KPI(activation_memory=1),
                                    1: KPI(activation_memory=2),
                                    0: KPI(activation_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_kpi_mapping)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=target_kpi)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_kpi = KPI(activation_memory=0)  # Infeasible solution!
        with self.assertRaises(Exception):
            bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                    target_kpi=target_kpi)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=KPI(activation_memory=np.inf))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 2)

    def test_search_weights_and_activation(self):
        target_kpi = KPI(weights_memory=2, activation_memory=2)
        layer_to_kpi_mapping = {0: {2: KPI(weights_memory=1, activation_memory=1),
                                    1: KPI(weights_memory=2, activation_memory=2),
                                    0: KPI(weights_memory=3, activation_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_kpi_mapping)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=target_kpi)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_kpi = KPI(weights_memory=0, activation_memory=0)  # Infeasible solution!
        with self.assertRaises(Exception):
            bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                    target_kpi=target_kpi)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=KPI(weights_memory=np.inf, activation_memory=np.inf))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 2)

    def test_search_total_kpi(self):
        target_kpi = KPI(total_memory=2)
        layer_to_kpi_mapping = {0: {2: KPI(total_memory=1),
                                    1: KPI(total_memory=2),
                                    0: KPI(total_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_kpi_mapping)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=target_kpi)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

    def test_search_bops_kpi(self):
        target_kpi = KPI(bops=2)
        layer_to_kpi_mapping = {0: {2: KPI(bops=1),
                                    1: KPI(bops=2),
                                    0: KPI(bops=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_kpi_mapping)

        bit_cfg = mp_integer_programming_search(mock_search_manager,
                                                target_kpi=target_kpi)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)


class TestSearchBitwidthConfiguration(unittest.TestCase):

    def test_search_engine(self):
        core_config = CoreConfig(n_iter=1, quantization_config=DEFAULTCONFIG,
                                 mixed_precision_config=MixedPrecisionQuantizationConfigV2(compute_mse,
                                                                                           get_average_weights,
                                                                                           num_of_images=1))

        base_config, mixed_precision_cfg_list = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(enable_activation_quantization=False)
        tp_model = generate_mixed_precision_test_tp_model(
            base_cfg=base_config,
            mp_bitwidth_candidates_list=[(c.weights_n_bits, c.activation_n_bits) for c in mixed_precision_cfg_list])
        tpc = generate_keras_tpc(name="bitwidth_cfg_test", tp_model=tp_model)
        fw_info = DEFAULT_KERAS_INFO
        in_model = MobileNetV2()
        keras_impl = KerasImplementation()

        def dummy_representative_dataset():
            return None

        graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading
        graph.set_fw_info(fw_info)
        graph.set_tpc(tpc)
        graph = set_quantization_configuration_to_graph(graph=graph,
                                                        quant_config=core_config.quantization_config,
                                                        mixed_precision_enable=core_config.mixed_precision_enable)

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
        keras_sens_eval = keras_impl.get_sensitivity_evaluator(graph,
                                                               core_config.mixed_precision_config,
                                                               representative_data_gen=lambda:
                                                               [np.random.random((1, 224, 224, 3))],
                                                               fw_info=fw_info)

        cfg = search_bit_width(graph_to_search_cfg=graph,
                               fw_info=DEFAULT_KERAS_INFO,
                               fw_impl=keras_impl,
                               target_kpi=KPI(np.inf),
                               mp_config=core_config.mixed_precision_config,
                               representative_data_gen=lambda: [np.random.random((1, 224, 224, 3))],
                               search_method=BitWidthSearchMethod.INTEGER_PROGRAMMING)

        with self.assertRaises(Exception):
            cfg = search_bit_width(graph_to_search_cfg=graph,
                                   fw_info=DEFAULT_KERAS_INFO,
                                   fw_impl=keras_impl,
                                   target_kpi=KPI(np.inf),
                                   mp_config=core_config.mixed_precision_config,
                                   representative_data_gen=lambda: [np.random.random((1, 224, 224, 3))],
                                   search_method=None)

        with self.assertRaises(Exception):
            cfg = search_bit_width(graph_to_search_cfg=graph,
                                   fw_info=DEFAULT_KERAS_INFO,
                                   fw_impl=keras_impl,
                                   target_kpi=None,
                                   mp_config=core_config.mixed_precision_config,
                                   representative_data_gen=lambda: [np.random.random((1, 224, 224, 3))],
                                   search_method=BitWidthSearchMethod.INTEGER_PROGRAMMING)


if __name__ == '__main__':
    unittest.main()
