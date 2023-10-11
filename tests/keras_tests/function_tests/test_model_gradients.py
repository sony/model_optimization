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

def multiple_outputs_node_model(input_shape):
    inputs = Input(shape=input_shape)
    x_conv = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization()(x_conv)
    x_relu = ReLU()(x_bn)
    x_split = tf.split(x_relu, num_or_size_splits=2, axis=-1)
    outputs = x_split[0]+x_split[1]
    # outputs = x_relu
    return keras.Model(inputs=inputs, outputs=outputs)


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

def _get_normalized_hessian_trace_approx(graph, interest_points, keras_impl, alpha, num_of_inputs=1):
    hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                        representative_dataset=functools.partial(representative_dataset, num_of_inputs=num_of_inputs),
                                                        fw_impl=keras_impl)
    x = []
    for interest_point in interest_points:
        request = hessian_common.TraceHessianRequest(mode=hessian_common.HessianMode.ACTIVATION,
                                                     granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                                     target_node=interest_point)
        hessian_data = hessian_service.fetch_hessian(request, 1)
        hessian_data_per_image = hessian_data[0]
        assert isinstance(hessian_data_per_image, list)
        assert len(hessian_data_per_image) == 1
        x.append(hessian_data_per_image[0])
    x = hessian_common.hessian_utils.normalize_weights(x, alpha=alpha, outputs_indices=[len(interest_points) - 1])
    return x


class TestModelGradients(unittest.TestCase):
    # TODO: change tests to ignore the normalization and check
    #  closeness to ACTUAL hessian values on small trained models.

    def _run_model_grad_test(self, graph, keras_impl, output_indices=None, num_of_inputs=1):
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]
        all_output_indices = [len(interest_points) - 1] if output_indices is None else output_indices
        x = _get_normalized_hessian_trace_approx(graph, interest_points, keras_impl, alpha=0.3, num_of_inputs=num_of_inputs)

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

        x = _get_normalized_hessian_trace_approx(graph, interest_points, keras_impl, alpha=0)

        # These are the expected values of the normalized gradients (gradients should be 2 and 1
        # with respect to input and mult layer, respectively)
        self.assertTrue(np.isclose(x[0], np.float32(0.8), 1e-1))
        self.assertTrue(np.isclose(x[1], np.float32(0.2), 1e-1))
        self.assertTrue(np.isclose(x[2], np.float32(0.0)))

        y = _get_normalized_hessian_trace_approx(graph, interest_points, keras_impl, alpha=1)

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


    def test_inputs_as_list_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = inputs_as_list_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset, generate_keras_tpc)
        self._run_model_grad_test(graph, keras_impl, num_of_inputs=2)

    def test_multiple_outputs_node(self):
        input_shape = (8, 8, 3)
        in_model = multiple_outputs_node_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model, keras_impl, DEFAULT_KERAS_INFO, representative_dataset,
                                           generate_keras_tpc)
        self._run_model_grad_test(graph, keras_impl)

