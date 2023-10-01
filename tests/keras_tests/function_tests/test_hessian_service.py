import unittest

import keras
import numpy as np
from tensorflow import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input

import model_compression_toolkit.core.common.hessian as hess
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
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)] * num_of_inputs


class TestHessianService(unittest.TestCase):

    def setUp(self):
        self.hessian_service = hess.hessian_service
        input_shape = (8, 8, 3)
        self.float_model = basic_model(input_shape)
        self.keras_impl = KerasImplementation()
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.keras_impl,
                                                DEFAULT_KERAS_INFO,
                                                representative_dataset,
                                                generate_keras_tpc)
        self.hessian_service.set_graph(self.graph)
        self.assertEqual(self.hessian_service.graph, self.graph)
        self.hessian_service.set_fw_impl(self.keras_impl)
        self.assertEqual(self.hessian_service.fw_impl, self.keras_impl)

    def test_add_hessian_configurations(self):
        self.hessian_service._set_hessian_configurations([])
        hessian_configs = [hess.HessianConfig(nodes_names_for_hessian_computation=[],
                                              granularity=hess.HessianGranularity.PER_LAYER,
                                              mode=hess.HessianMode.ACTIVATIONS)]
        self.hessian_service.add_hessian_configurations(hessian_configs)
        self.assertEqual(self.hessian_service._hessian_configurations, hessian_configs)

    def test_reused_hessian_computation(self):
        self.hessian_service.clear_cache()

        config1 = hess.HessianConfig(
            mode=hess.HessianMode.ACTIVATIONS,
            granularity=hess.HessianGranularity.PER_LAYER,
            nodes_names_for_hessian_computation=list(self.graph.nodes),
            alpha=0.5,
            num_iterations=10
        )

        config2 = hess.HessianConfig(
            mode=hess.HessianMode.ACTIVATIONS,
            granularity=hess.HessianGranularity.PER_LAYER,
            nodes_names_for_hessian_computation=list(self.graph.nodes),
            alpha=0.5,
            num_iterations=10
        )

        images = next(representative_dataset())
        hessian_data1 = self.hessian_service.fetch_hessian(config1, images)
        self.assertTrue(len(self.hessian_service.fetch_hessian(config1)) == 1)
        hessian_data2 = self.hessian_service.fetch_hessian(config2, images)
        self.assertTrue(len(self.hessian_service.fetch_hessian(config2)) == 1)
        self.assertEqual(hessian_data1, hessian_data2)
        num_hessians = self.hessian_service._count_cache()
        self.assertTrue(num_hessians == 1)

    def test_double_hessian_computation(self):
        self.hessian_service.clear_cache()

        config1 = hess.HessianConfig(
            mode=hess.HessianMode.ACTIVATIONS,
            granularity=hess.HessianGranularity.PER_LAYER,
            nodes_names_for_hessian_computation=list(self.graph.nodes),
            alpha=0.5,
            num_iterations=10
        )

        config2 = hess.HessianConfig(
            mode=hess.HessianMode.ACTIVATIONS,
            granularity=hess.HessianGranularity.PER_LAYER,
            nodes_names_for_hessian_computation=list(self.graph.nodes),
            alpha=0.3,
            num_iterations=10
        )

        images = next(representative_dataset())
        self.hessian_service.fetch_hessian(config1, images)
        self.assertTrue(len(self.hessian_service.fetch_hessian(config1)) == 1)
        self.hessian_service.fetch_hessian(config2, images)
        self.assertTrue(len(self.hessian_service.fetch_hessian(config2)) == 1)
        num_hessians = self.hessian_service._count_cache()
        self.assertTrue(num_hessians == 2)


if __name__ == "__main__":
    unittest.main()
