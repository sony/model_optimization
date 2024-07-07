# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import keras
import numpy as np
from tensorflow import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input

from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianScoresRequest, HessianMode, \
    HessianScoresGranularity
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs


def basic_model(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def multiple_act_nodes_model(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    x_relu = ReLU()(x_bn)
    outputs = Conv2D(2, 3, padding='same', name="conv2d_2")(x_relu)
    return keras.Model(inputs=inputs, outputs=outputs)


def representative_dataset():
    for _ in range(2):
        yield [np.random.randn(2, 8, 8, 3).astype(np.float32)]


class TestHessianService(unittest.TestCase):
    def tearDown(self) -> None:
        del self.hessian_service

    def setUp(self):

        input_shape = (8, 8, 3)
        self.float_model = basic_model(input_shape)
        self.keras_impl = KerasImplementation()
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.keras_impl,
                                                DEFAULT_KERAS_INFO,
                                                representative_dataset,
                                                generate_keras_tpc)

        self.hessian_service = HessianInfoService(graph=self.graph, representative_dataset_gen=representative_dataset,
                                                  fw_impl=self.keras_impl)

        self.assertEqual(self.hessian_service.graph, self.graph)
        self.assertEqual(self.hessian_service.fw_impl, self.keras_impl)

    def test_fetch_activation_hessian(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]])
        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(len(hessian[0]), 2, "Expecting 2 Hessian scores.")

    def test_fetch_weights_hessian(self):
        request = HessianScoresRequest(mode=HessianMode.WEIGHTS,
                                       granularity=HessianScoresGranularity.PER_OUTPUT_CHANNEL,
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[1]])
        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(len(hessian[0]), 2, "Expecting 2 Hessian scores.")

    def test_fetch_not_enough_samples_throw(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]])

        with self.assertRaises(Exception) as e:
            hessian = self.hessian_service.fetch_hessian(request, 5, batch_size=2)  # representative dataset produces 4 images total

        self.assertTrue('Not enough samples in the provided representative dataset' in str(e.exception))

    def test_fetch_not_enough_samples_small_batch_throw(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]])

        with self.assertRaises(Exception) as e:
            hessian = self.hessian_service.fetch_hessian(request, 5, batch_size=1)  # representative dataset produces 4 images total

        self.assertTrue('Not enough samples in the provided representative dataset' in str(e.exception))

    def test_fetch_compute_batch_larger_than_repr_batch(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]])

        hessian = self.hessian_service.fetch_hessian(request, 3, batch_size=3)  # representative batch size is 2
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(len(hessian[0]), 3, "Expecting 3 Hessian scores.")

    def test_fetch_required_zero(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]], )

        hessian = self.hessian_service.fetch_hessian(request, 0)

        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(len(hessian[0]), 0, "Expecting an empty Hessian scores list.")

    def test_fetch_multiple_nodes(self):
        input_shape = (8, 8, 3)
        self.float_model = multiple_act_nodes_model(input_shape)
        self.keras_impl = KerasImplementation()
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.keras_impl,
                                                DEFAULT_KERAS_INFO,
                                                representative_dataset,
                                                generate_keras_tpc)

        self.hessian_service = HessianInfoService(graph=self.graph,
                                                  representative_dataset_gen=representative_dataset,
                                                  fw_impl=self.keras_impl)

        self.assertEqual(self.hessian_service.graph, self.graph)
        self.assertEqual(self.hessian_service.fw_impl, self.keras_impl)

        graph_nodes = list(self.graph.get_topo_sorted_nodes())
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[graph_nodes[0], graph_nodes[2]])

        hessian = self.hessian_service.fetch_hessian(request, 2)

        self.assertEqual(len(hessian), 2, "Expecting returned Hessian list to include two list of "
                                          "approximation, for the two target nodes.")
        self.assertEqual(len(hessian[0]), 2, f"Expecting 2 Hessian scores for layer {graph_nodes[0].name}.")
        self.assertEqual(len(hessian[1]), 2, f"Expecting 2 Hessian scores for layer {graph_nodes[2].name}.")

    def test_clear_cache(self):
        self.hessian_service._clear_saved_hessian_info()
        target_node = list(self.graph.nodes)[1]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[target_node])
        self.assertEqual(self.hessian_service.count_saved_scores_of_request(request)[target_node], 0)

        self.hessian_service.fetch_hessian(request, 1)
        self.assertEqual(self.hessian_service.count_saved_scores_of_request(request)[target_node], 1)
        self.hessian_service._clear_saved_hessian_info()
        self.assertEqual(self.hessian_service.count_saved_scores_of_request(request)[target_node], 0)

    def test_double_fetch_hessian(self):
        self.hessian_service._clear_saved_hessian_info()
        target_node = list(self.graph.nodes)[1]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[target_node])
        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(len(hessian[0]), 2, "Expecting 2 Hessian scores.")
        self.assertEqual(self.hessian_service.count_saved_scores_of_request(request)[target_node], 2)

        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(len(hessian[0]), 2, "Expecting 2 Hessian scores.")
        self.assertEqual(self.hessian_service.count_saved_scores_of_request(request)[target_node], 2)

    def test_populate_cache_to_size(self):
        self.hessian_service._clear_saved_hessian_info()
        target_node = list(self.graph.nodes)[1]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=[target_node])
        self.hessian_service._populate_saved_info_to_size(request, 2)
        self.assertEqual(self.hessian_service.count_saved_scores_of_request(request)[target_node], 2)


if __name__ == "__main__":
    unittest.main()
