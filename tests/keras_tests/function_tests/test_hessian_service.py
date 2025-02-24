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
from model_compression_toolkit.core.keras.data_util import data_gen_to_dataloader
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs


def basic_model(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def multiple_act_nodes_model(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    x_relu = ReLU()(x_bn)
    outputs = Conv2D(2, 3, padding='same', name="conv2d_2")(x_relu)
    return keras.Model(inputs=inputs, outputs=outputs)


def get_representative_dataset_fn(n_iters=10):
    def f():
        for _ in range(n_iters):
            yield [np.random.randn(2, 16, 16, 3).astype(np.float32)]
    return f


class TestHessianService(unittest.TestCase):
    def tearDown(self) -> None:
        del self.hessian_service

    def setUp(self):

        input_shape = (16, 16, 3)
        self.float_model = basic_model(input_shape)
        self.keras_impl = KerasImplementation()
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.keras_impl,
                                                DEFAULT_KERAS_INFO,
                                                get_representative_dataset_fn(),
                                                generate_keras_tpc,
                                                attach2fw=AttachTpcToKeras())

        self.hessian_service = HessianInfoService(graph=self.graph, fw_impl=self.keras_impl)

        self.assertEqual(self.hessian_service.graph, self.graph)
        self.assertEqual(self.hessian_service.fw_impl, self.keras_impl)

    def test_fetch_activation_hessian(self):
        node = list(self.graph.get_topo_sorted_nodes())[0]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=2,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[node])
        hessian = self.hessian_service.fetch_hessian(request)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(hessian[node.name].shape[0], 2, "Expecting 2 Hessian scores.")

    def test_fetch_weights_hessian(self):
        node = list(self.graph.get_topo_sorted_nodes())[1]
        request = HessianScoresRequest(mode=HessianMode.WEIGHTS,
                                       granularity=HessianScoresGranularity.PER_OUTPUT_CHANNEL,
                                       n_samples=2,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[node])
        hessian = self.hessian_service.fetch_hessian(request)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(hessian[node.name].shape[0], 2, "Expecting 2 Hessian scores.")

    def test_fetch_not_enough_samples_throw(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=5,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(2), batch_size=2),
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]])

        with self.assertRaises(Exception) as e:
            self.hessian_service.fetch_hessian(request)  # representative dataset produces 4 images total

        self.assertTrue('not enough samples in the provided representative dataset' in str(e.exception))

    def test_fetch_not_enough_samples_small_batch_throw(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=5,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(2), batch_size=1),
                                       target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]])

        with self.assertRaises(Exception) as e:
            hessian = self.hessian_service.fetch_hessian(request)  # representative dataset produces 4 images total

        self.assertTrue('not enough samples in the provided representative dataset' in str(e.exception))

    def test_fetch_compute_batch_larger_than_repr_batch(self):
        node = list(self.graph.get_topo_sorted_nodes())[0]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=3,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=3),
                                       target_nodes=[node])

        hessian = self.hessian_service.fetch_hessian(request)  # representative batch size is 2
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(hessian[node.name].shape[0], 3, "Expecting 3 Hessian scores.")

    def test_fetch_required_zero(self):
        node = list(self.graph.get_topo_sorted_nodes())[0]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=0,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[node])

        hessian = self.hessian_service.fetch_hessian(request)

        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(hessian[node.name].shape[0], 0, "Expecting an empty Hessian scores list.")

    def test_fetch_multiple_nodes(self):
        input_shape = (16, 16, 3)
        self.float_model = multiple_act_nodes_model(input_shape)
        self.keras_impl = KerasImplementation()
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.keras_impl,
                                                DEFAULT_KERAS_INFO,
                                                get_representative_dataset_fn(),
                                                generate_keras_tpc,
                                                attach2fw=AttachTpcToKeras())

        self.hessian_service = HessianInfoService(graph=self.graph, fw_impl=self.keras_impl)

        self.assertEqual(self.hessian_service.graph, self.graph)
        self.assertEqual(self.hessian_service.fw_impl, self.keras_impl)

        graph_nodes = list(self.graph.get_topo_sorted_nodes())
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=2,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[graph_nodes[0], graph_nodes[2]])

        hessian = self.hessian_service.fetch_hessian(request)

        self.assertEqual(len(hessian), 2, "Expecting returned Hessian list to include two list of "
                                          "approximation, for the two target nodes.")
        self.assertEqual(hessian[graph_nodes[0].name].shape[0], 2, f"Expecting 2 Hessian scores for layer {graph_nodes[0].name}.")
        self.assertEqual(hessian[graph_nodes[2].name].shape[0], 2, f"Expecting 2 Hessian scores for layer {graph_nodes[2].name}.")

    def test_clear_cache(self):
        self.hessian_service.clear_cache()
        self.assertTrue(len(self.hessian_service.cache._data) == 0)

    def test_double_fetch_hessian(self):
        self.hessian_service.clear_cache()
        target_node = list(self.graph.nodes)[1]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=2,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[target_node])
        hessian = self.hessian_service.fetch_hessian(request)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(hessian[target_node.name].shape[0], 2, "Expecting 2 Hessian scores.")

        # can fetch second time from cache
        request = request.clone(data_loader=None)
        hessian = self.hessian_service.fetch_hessian(request)
        self.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.assertEqual(hessian[target_node.name].shape[0], 2, "Expecting 2 Hessian scores.")

    def test_invalid_request(self):
        with self.assertRaises(Exception, msg='Data loader and the number of samples cannot both be None'):
            HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                 granularity=HessianScoresGranularity.PER_TENSOR,
                                 n_samples=None,
                                 data_loader=None,
                                 target_nodes=list(self.graph.nodes))

    def test_fetch_hessian_invalid_args(self):
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=None,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=list(self.graph.nodes))
        with self.assertRaises(Exception, msg='Number of samples can be None only when force_compute is True.'):
            self.hessian_service.fetch_hessian(request)

    def test_double_fetch_more_samples(self):
        # this is mostly for coverage
        self.hessian_service.clear_cache()
        node = list(self.graph.get_topo_sorted_nodes())[0]
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=2,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[node])
        hess = self.hessian_service.fetch_hessian(request)
        assert hess[node.name].shape[0] == 2

        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       n_samples=4,
                                       data_loader=data_gen_to_dataloader(get_representative_dataset_fn(), batch_size=1),
                                       target_nodes=[node])
        hess = self.hessian_service.fetch_hessian(request)
        assert hess[node.name].shape[0] == 4


if __name__ == "__main__":
    unittest.main()
