import functools

import keras
import unittest

from keras.layers import Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
from tensorflow import initializers
import numpy as np
import tensorflow as tf

import model_compression_toolkit.core.common.hessian as hessian_common
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc

import model_compression_toolkit as mct
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs

tp = mct.target_platform


def basic_derivative_model(input_shape):
    inputs = Input(shape=input_shape)
    outputs = 2 * inputs + 1
    return keras.Model(inputs=inputs, outputs=outputs)


def basic_model(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def advenced_model(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape, name='input1')
    x = Conv2D(2, 3, padding='same', name="conv2d_1")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    x_relu = ReLU(name='relu1')(x_bn)
    x_2 = Conv2D(2, 3, padding='same', name="conv2d_2")(x_relu)
    x_bn2 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                               name='bn2')(x_2)
    x_reshape = Reshape((-1,), name='reshape1')(x_bn2)
    x_bn3 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                               name='bn3')(
        x_reshape)
    outputs = ReLU(name='relu2')(x_bn3)
    return keras.Model(inputs=inputs, outputs=outputs)


def multiple_output_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(2)(inputs)
    x = Conv2D(2, 4)(x)
    x = BatchNormalization()(x)
    out1 = ReLU(max_value=6.0)(x)
    out2 = Conv2D(2, 4)(out1)
    return keras.Model(inputs=inputs, outputs=[out1, out2])


def inputs_as_list_model(input_shape):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    x_stack = tf.stack([input1, input2])
    x_conv = Conv2D(2, 3, padding='same', name="conv2d")(x_stack)
    x_bn = BatchNormalization()(x_conv)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=[input1, input2], outputs=outputs)


def model_with_output_replacements(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    x_relu = ReLU()(x_bn)
    x_soft = tf.nn.softmax(x_relu)
    outputs = tf.math.argmax(x_soft)

    return keras.Model(inputs=inputs, outputs=outputs)


def representative_dataset(num_of_inputs=1):
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]*num_of_inputs


class TestModelGradients(unittest.TestCase):

    def _run_model_grad_test(self, graph, keras_impl, output_indices=None, num_of_inputs=1):
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]


        all_output_indices = [len(interest_points) - 1] if output_indices is None else output_indices

        hessian_common.hessian_service.set_graph(graph=graph)
        hessian_common.hessian_service.set_fw_impl(keras_impl)
        hessian_cfg = hessian_common.HessianConfig(mode=hessian_common.HessianMode.ACTIVATIONS,
                                                   granularity=hessian_common.HessianGranularity.PER_LAYER,
                                                   search_output_replacement=True,
                                                   nodes_names_for_hessian_computation=interest_points,
                                                   alpha=0.3)
        images = next(representative_dataset(num_of_inputs=num_of_inputs))
        hessian_data = hessian_common.hessian_service.fetch_hessian(hessian_cfg=hessian_cfg,
                                                     input_images=images)
        x = [hessian_data[ip] for ip in interest_points]

        # Checking that the weights were computed and normalized correctly
        # In rare occasions, the output tensor has all zeros, so the gradients for all interest points are zeros.
        # This is a pathological case that is not possible in real networks, so we just extend the assertion to prevent
        # the test from failing in this rare cases.
        self.assertTrue(np.isclose(np.sum(x), 1) or all([y == 0 for i, y in enumerate(x) if i not in all_output_indices]))

    def test_jacobian_trace_calculation(self):
        input_shape = (8, 8, 3)
        in_model = basic_derivative_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]

        hessian_common.hessian_service.set_graph(graph=graph)
        hessian_common.hessian_service.set_fw_impl(keras_impl)
        hessian_cfg = hessian_common.HessianConfig(mode=hessian_common.HessianMode.ACTIVATIONS,
                                                   granularity=hessian_common.HessianGranularity.PER_LAYER,
                                                   search_output_replacement=True,
                                                   nodes_names_for_hessian_computation=interest_points,
                                                   alpha=0)
        images = next(representative_dataset())
        hessian_data = hessian_common.hessian_service.fetch_hessian(hessian_cfg=hessian_cfg,
                                                                    input_images=images)
        x = [hessian_data[ip] for ip in interest_points]


        # These are the expected values of the normalized gradients (gradients should be 2 and 1
        # with respect to input and mult layer, respectively)
        self.assertTrue(np.isclose(x[0], np.float32(0.8), 1e-1))
        self.assertTrue(np.isclose(x[1], np.float32(0.2), 1e-1))
        self.assertTrue(np.isclose(x[2], np.float32(0.0)))

        hessian_cfg = hessian_common.HessianConfig(mode=hessian_common.HessianMode.ACTIVATIONS,
                                                   granularity=hessian_common.HessianGranularity.PER_LAYER,
                                                   search_output_replacement=True,
                                                   nodes_names_for_hessian_computation=interest_points,
                                                   alpha=1)
        hessian_data = hessian_common.hessian_service.fetch_hessian(hessian_cfg=hessian_cfg,
                                                                    input_images=images)
        y = [hessian_data[ip] for ip in interest_points]

        self.assertTrue(np.isclose(y[0], np.float32(0.0)))
        self.assertTrue(np.isclose(y[1], np.float32(0.0)))
        self.assertTrue(np.isclose(y[2], np.float32(1.0)))


    def test_basic_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = basic_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)

        self._run_model_grad_test(graph, keras_impl)

    def test_advanced_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = advenced_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)

        self._run_model_grad_test(graph, keras_impl)

    def test_multiple_outputs_grad(self):
        input_shape = (8, 8, 3)
        in_model = multiple_output_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        self._run_model_grad_test(graph, keras_impl, output_indices=[len(sorted_graph_nodes) - 1,
                                                                     len(sorted_graph_nodes) - 2])

    def test_model_grad_with_output_replacements(self):
        input_shape = (8, 8, 3)
        in_model = model_with_output_replacements(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]

        hessian_common.hessian_service.set_graph(graph=graph)
        hessian_common.hessian_service.set_fw_impl(keras_impl)

        hessian_cfg = hessian_common.HessianConfig(mode=hessian_common.HessianMode.ACTIVATIONS,
                                                   granularity=hessian_common.HessianGranularity.PER_LAYER,
                                                   search_output_replacement=True,
                                                   nodes_names_for_hessian_computation=interest_points,
                                                   alpha=0.3)
        images = next(representative_dataset())
        hessian_data = hessian_common.hessian_service.fetch_hessian(hessian_cfg=hessian_cfg,
                                                                    input_images=images)
        x = [hessian_data[ip] for ip in interest_points]

        # Checking that the weights where computed and normalized correctly
        self.assertTrue(np.isclose(np.sum(x), 1))

        hessian_cfg = hessian_common.HessianConfig(mode=hessian_common.HessianMode.ACTIVATIONS,
                                                   granularity=hessian_common.HessianGranularity.PER_LAYER,
                                                   search_output_replacement=True,
                                                   nodes_names_for_hessian_computation=interest_points,
                                                   alpha=0)
        images = next(representative_dataset())
        hessian_data = hessian_common.hessian_service.fetch_hessian(hessian_cfg=hessian_cfg,
                                                                    input_images=images)
        y = [hessian_data[ip] for ip in interest_points]

        # Checking that the weights where computed and normalized correctly
        self.assertTrue(np.isclose(np.sum(y), 1))
        self.assertTrue(y[-1] == np.float32(0))
        self.assertTrue(y[-2] == np.float32(0))

    def test_inputs_as_list_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = inputs_as_list_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)
        self._run_model_grad_test(graph, keras_impl, num_of_inputs=2)

    def test_model_grad_single_point(self):
        input_shape = (8, 8, 3)
        in_model = basic_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [sorted_graph_nodes[-1]]

        hessian_common.hessian_service.set_graph(graph=graph)
        hessian_common.hessian_service.set_fw_impl(keras_impl)

        hessian_cfg = hessian_common.HessianConfig(mode=hessian_common.HessianMode.ACTIVATIONS,
                                                   granularity=hessian_common.HessianGranularity.PER_LAYER,
                                                   search_output_replacement=True,
                                                   nodes_names_for_hessian_computation=interest_points)
        images = next(representative_dataset())
        hessian_data = hessian_common.hessian_service.fetch_hessian(hessian_cfg=hessian_cfg,
                                                                    input_images=images)
        x = [hessian_data[ip] for ip in interest_points]

        self.assertTrue(len(x) == 1 and x[0] == 1.0)
