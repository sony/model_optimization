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
from unittest.mock import Mock

import numpy as np
import unittest

import keras
from model_compression_toolkit.core import DEFAULTCONFIG
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization, RUTarget
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width, \
    BitWidthSearchMethod
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    MixedPrecisionIntegerLPSolver
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras


class MockReconstructionHelper:
    def __init__(self):
        pass

    def reconstruct_config_from_virtual_graph(self,
                                              max_ru_config,
                                              changed_virtual_nodes_idx=None,
                                              original_base_config=None):
        return max_ru_config


class MockMixedPrecisionSearchManager:
    def __init__(self, layer_to_ru_mapping, ru_targets):
        self.ru_targets = ru_targets
        self.layer_to_bitwidth_mapping = {0: [0, 1, 2]}
        self.layer_to_ru_mapping = layer_to_ru_mapping
        self.min_ru = {t: np.array([1]) for t in ru_targets} # minimal resource utilization in the tests layer_to_ru_mapping

        self.max_ru_config = [0]
        self.config_reconstruction_helper = MockReconstructionHelper()

    def build_sensitivity_mapping(self):
        return {0: {0: 0, 1: 1, 2: 2}}

    def compute_resource_utilization_matrices(self):
        # minus 1 is normalization by the minimal resource utilization (which is always 1 in this test)
        ru = {
            RUTarget.WEIGHTS:
                [np.flip(np.array([ru.weights_memory - 1 for _, ru in self.layer_to_ru_mapping[0].items()]))],
            RUTarget.ACTIVATION:
                [np.flip(np.array([ru.activation_memory - 1 for _, ru in self.layer_to_ru_mapping[0].items()]))],
            RUTarget.BOPS:
                [np.flip(np.array([ru.bops - 1 for _, ru in self.layer_to_ru_mapping[0].items()]))],
            RUTarget.TOTAL:
                [np.flip(np.array([ru.total_memory - 1 for _, ru in self.layer_to_ru_mapping[0].items()]))]
        }
        return {k: np.array(v).T for k, v in ru.items() if k in self.ru_targets}


class TestLpSearchBitwidth(unittest.TestCase):

    def _execute(self, mock_search_mgr, target_resource_utilization):
        candidates_sensitivity = mock_search_mgr.build_sensitivity_mapping()
        candidates_ru = mock_search_mgr.compute_resource_utilization_matrices()
        min_ru = mock_search_mgr.min_ru
        ru_constraints = {k: v - min_ru[k] for k, v in target_resource_utilization.get_resource_utilization_dict(restricted_only=True).items()}
        lp_solver = MixedPrecisionIntegerLPSolver(candidates_sensitivity, candidates_ru, ru_constraints)
        return lp_solver.run()

    def test_search_weights_only(self):
        target_resource_utilization = ResourceUtilization(weights_memory=2)
        layer_to_ru_mapping = {0: {2: ResourceUtilization(weights_memory=1),
                                   1: ResourceUtilization(weights_memory=2),
                                   0: ResourceUtilization(weights_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_ru_mapping, {RUTarget.WEIGHTS})
        bit_cfg = self._execute(mock_search_manager, target_resource_utilization)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_resource_utilization = ResourceUtilization(weights_memory=0)  # Infeasible solution!
        with self.assertRaises(Exception):
            self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        bit_cfg = self._execute(mock_search_manager, target_resource_utilization=ResourceUtilization(weights_memory=1000))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 0)  # expecting for the maximal bit-width result

        target_resource_utilization = None  # target ResourceUtilization is not defined!
        with self.assertRaises(Exception):
            self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        with self.assertRaises(Exception):
            self._execute(mock_search_manager, target_resource_utilization=ResourceUtilization(weights_memory=np.inf))

    def test_search_activation_only(self):
        target_resource_utilization = ResourceUtilization(activation_memory=2)
        layer_to_ru_mapping = {0: {2: ResourceUtilization(activation_memory=1),
                                   1: ResourceUtilization(activation_memory=2),
                                   0: ResourceUtilization(activation_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_ru_mapping, {RUTarget.ACTIVATION})

        bit_cfg = self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_resource_utilization = ResourceUtilization(activation_memory=0)  # Infeasible solution!
        with self.assertRaises(Exception):
            bit_cfg = self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        bit_cfg = self._execute(mock_search_manager,
                                target_resource_utilization=ResourceUtilization(activation_memory=1000))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 0)  # expecting for the maximal bit-width result

    def test_search_weights_and_activation(self):
        target_resource_utilization = ResourceUtilization(weights_memory=2, activation_memory=2)
        layer_to_ru_mapping = {0: {2: ResourceUtilization(weights_memory=1, activation_memory=1),
                                   1: ResourceUtilization(weights_memory=2, activation_memory=2),
                                   0: ResourceUtilization(weights_memory=3, activation_memory=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_ru_mapping, {RUTarget.WEIGHTS, RUTarget.ACTIVATION})

        bit_cfg = self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

        target_resource_utilization = ResourceUtilization(weights_memory=0, activation_memory=0)  # Infeasible solution!
        with self.assertRaises(Exception):
            bit_cfg = self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        bit_cfg = self._execute(mock_search_manager, target_resource_utilization=ResourceUtilization(weights_memory=1000,
                                                                                                activation_memory=1000))

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 0)  # expecting for the maximal bit-width result

    def test_search_total_resource_utilization(self):
        target_resource_utilization = ResourceUtilization(total_memory=4)
        layer_to_ru_mapping = {0: {2: ResourceUtilization(weights_memory=1, activation_memory=1, total_memory=2),
                                   1: ResourceUtilization(weights_memory=2, activation_memory=2, total_memory=4),
                                   0: ResourceUtilization(weights_memory=3, activation_memory=3, total_memory=6)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_ru_mapping, {RUTarget.TOTAL})

        bit_cfg = self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)

    def test_search_bops_ru(self):
        target_resource_utilization = ResourceUtilization(bops=2)
        layer_to_ru_mapping = {0: {2: ResourceUtilization(bops=1),
                                   1: ResourceUtilization(bops=2),
                                   0: ResourceUtilization(bops=3)}}
        mock_search_manager = MockMixedPrecisionSearchManager(layer_to_ru_mapping, {RUTarget.BOPS})

        bit_cfg = self._execute(mock_search_manager, target_resource_utilization=target_resource_utilization)

        self.assertTrue(len(bit_cfg) == 1)
        self.assertTrue(bit_cfg[0] == 1)


class TestSearchBitwidthConfiguration(unittest.TestCase):

    def run_search_bitwidth_config_test(self, core_config):
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(enable_activation_quantization=False)

        tpc = get_weights_only_mp_tpc_keras(base_config=base_config,
                                            default_config=default_config,
                                            mp_bitwidth_candidates_list=[
                                                (c.attr_weights_configs_mapping[KERNEL_ATTR].weights_n_bits,
                                                 c.activation_n_bits) for c
                                                in mixed_precision_cfg_list],
                                            name="bitwidth_cfg_test")

        fw_info = DEFAULT_KERAS_INFO
        input_shape = (1, 8, 8, 3)
        input_tensor = keras.layers.Input(shape=input_shape[1:])
        conv = keras.layers.Conv2D(3, 3)(input_tensor)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.ReLU()(bn)
        in_model = keras.Model(inputs=input_tensor, outputs=relu)
        keras_impl = KerasImplementation()

        def dummy_representative_dataset():
            return None

        graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading

        fqc = AttachTpcToKeras().attach(tpc)

        graph.set_fw_info(fw_info)
        graph.set_fqc(fqc)
        graph = set_quantization_configuration_to_graph(graph=graph,
                                                        quant_config=core_config.quantization_config,
                                                        mixed_precision_enable=True)

        for node in graph.nodes:
            node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                             fw_info=fw_info,
                                                             graph=graph)

        mi = ModelCollector(graph,
                            fw_info=DEFAULT_KERAS_INFO,
                            fw_impl=keras_impl,
                            qc=core_config.quantization_config)

        for i in range(1):
            mi.infer([np.random.randn(*input_shape)])

        def representative_data_gen():
            yield [np.random.random(input_shape)]

        calculate_quantization_params(graph, fw_impl=keras_impl, repr_data_gen_fn=representative_data_gen)

        keras_impl.get_sensitivity_evaluator(graph,
                                             core_config.mixed_precision_config,
                                             representative_data_gen,
                                             fw_info=fw_info)

        cfg = search_bit_width(graph=graph,
                               fw_info=DEFAULT_KERAS_INFO,
                               fw_impl=keras_impl,
                               target_resource_utilization=ResourceUtilization(weights_memory=100),
                               mp_config=core_config.mixed_precision_config,
                               representative_data_gen=representative_data_gen,
                               search_method=BitWidthSearchMethod.INTEGER_PROGRAMMING)

    def test_mixed_precision_search_facade(self):
        core_config_avg_weights = CoreConfig(quantization_config=DEFAULTCONFIG,
                                             mixed_precision_config=MixedPrecisionQuantizationConfig(compute_mse,
                                                                                                     MpDistanceWeighting.AVG,
                                                                                                     num_of_images=1,
                                                                                                     use_hessian_based_scores=False))

        self.run_search_bitwidth_config_test(core_config_avg_weights)

        core_config_last_layer = CoreConfig(quantization_config=DEFAULTCONFIG,
                                            mixed_precision_config=MixedPrecisionQuantizationConfig(compute_mse,
                                                                                                    MpDistanceWeighting.LAST_LAYER,
                                                                                                    num_of_images=1,
                                                                                                    use_hessian_based_scores=False))

        self.run_search_bitwidth_config_test(core_config_last_layer)


if __name__ == '__main__':
    unittest.main()
