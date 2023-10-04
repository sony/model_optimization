import unittest

import keras
import numpy as np
from tensorflow import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input

import model_compression_toolkit.core.common.hessian as hess
from model_compression_toolkit.core.common.hessian import HessianService, HessianConfig, HessianRequest, HessianMode, \
    HessianGranularity
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


def representative_dataset(num_of_inputs=1):
    yield [np.random.randn(2, 8, 8, 3).astype(np.float32)] * num_of_inputs


class TestHessianService(unittest.TestCase):

    def setUp(self):

        input_shape = (8, 8, 3)
        self.float_model = basic_model(input_shape)
        self.keras_impl = KerasImplementation()
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.keras_impl,
                                                DEFAULT_KERAS_INFO,
                                                representative_dataset,
                                                generate_keras_tpc)
        hessian_cfg = HessianConfig()
        self.hessian_service = HessianService(graph=self.graph,
                                              representative_dataset=representative_dataset,
                                              fw_impl=self.keras_impl,
                                              hessian_configuration=hessian_cfg)

        self.assertEqual(self.hessian_service.graph, self.graph)
        self.assertEqual(self.hessian_service.fw_impl, self.keras_impl)

    def test_fetch_hessian(self):
        request = HessianRequest(mode=HessianMode.ACTIVATIONS,
                                 granularity=HessianGranularity.PER_TENSOR,
                                 target_node=list(self.graph.nodes)[1])
        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 2)

    def test_clear_cache(self):
        self.hessian_service.clear_cache()
        request = HessianRequest(mode=HessianMode.ACTIVATIONS,
                                 granularity=HessianGranularity.PER_TENSOR,
                                 target_node=list(self.graph.nodes)[1])
        self.assertEqual(self.hessian_service.count_cache_of_request(request), 0)

        self.hessian_service.fetch_hessian(request, 1)
        self.assertEqual(self.hessian_service.count_cache_of_request(request), 1)
        self.hessian_service.clear_cache()
        self.assertEqual(self.hessian_service.count_cache_of_request(request), 0)


    def test_double_fetch_hessian(self):
        self.hessian_service.clear_cache()
        request = HessianRequest(mode=HessianMode.ACTIVATIONS,
                                 granularity=HessianGranularity.PER_TENSOR,
                                 target_node=list(self.graph.nodes)[1])
        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 2)
        self.assertEqual(self.hessian_service.count_cache_of_request(request), 2)

        hessian = self.hessian_service.fetch_hessian(request, 2)
        self.assertEqual(len(hessian), 2)
        self.assertEqual(self.hessian_service.count_cache_of_request(request), 2)

    def test_populate_cache_to_size(self):
        self.hessian_service.clear_cache()
        request = HessianRequest(mode=HessianMode.ACTIVATIONS,
                                 granularity=HessianGranularity.PER_TENSOR,
                                 target_node=list(self.graph.nodes)[1])
        self.hessian_service._populate_cache_to_size(request, 2)
        self.assertEqual(self.hessian_service.count_cache_of_request(request), 2)


if __name__ == "__main__":
    unittest.main()
