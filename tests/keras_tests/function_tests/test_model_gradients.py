import keras
import unittest

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
from tensorflow import initializers
import numpy as np
import tensorflow as tf

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_tp_model, \
    get_op_quantization_configs
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc

import model_compression_toolkit as mct
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


def representative_dataset():
    return [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def prepare_graph(in_model, keras_impl):
    fw_info = DEFAULT_KERAS_INFO
    qc = DEFAULTCONFIG

    graph = keras_impl.model_reader(in_model, representative_dataset)  # model reading
    graph = substitute(graph, keras_impl.get_substitutions_prepare_graph())
    for node in graph.nodes:
        node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                         fw_info=fw_info, graph=graph)
    graph = substitute(graph, keras_impl.get_substitutions_pre_statistics_collection(qc))

    base_config, op_cfg_list = get_op_quantization_configs()
    tp = generate_tp_model(base_config, base_config, op_cfg_list, "model_grad_test")
    tpc = generate_keras_tpc(name="model_grad_test", tp_model=tp)

    graph.set_fw_info(fw_info)
    graph.set_tpc(tpc)

    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=qc)
    return graph


class TestModelGradients(unittest.TestCase):

    def _run_model_grad_test(self, graph, keras_impl):
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]

        input_tensors = {inode: representative_dataset()[0] for inode in graph.get_inputs()}
        output_nodes = [o.node for o in graph.output_nodes]

        x = keras_impl.model_grad(graph_float=graph,
                                  model_input_tensors=input_tensors,
                                  interest_points=interest_points,
                                  output_list=output_nodes,
                                  all_outputs_indices=[len(interest_points) - 1],
                                  alpha=0.3)

        # Checking that the wiehgts where computed and normalized correctly
        self.assertTrue(np.isclose(np.sum(x), 1))

    def test_jacobian_trace_calculation(self):
        input_shape = (8, 8, 3)
        in_model = basic_derivative_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph(in_model, keras_impl)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]

        input_tensors = {inode: representative_dataset()[0] for inode in graph.get_inputs()}
        output_nodes = [o.node for o in graph.output_nodes]
        x = keras_impl.model_grad(graph_float=graph,
                                  model_input_tensors=input_tensors,
                                  interest_points=interest_points,
                                  output_list=output_nodes,
                                  all_outputs_indices=[len(interest_points) - 1],
                                  alpha=0)

        # These are the expected values of the normalized gradients (gradients should be 2 and 1
        # with respect to input and mult layer, respectively)
        self.assertTrue(np.isclose(x[0], np.float32(0.66), 1e-1))
        self.assertTrue(np.isclose(x[1], np.float32(0.33), 1e-1))
        self.assertTrue(np.isclose(x[2], np.float32(0.0)))

        y = keras_impl.model_grad(graph_float=graph,
                                  model_input_tensors=input_tensors,
                                  interest_points=interest_points,
                                  output_list=output_nodes,
                                  all_outputs_indices=[len(interest_points) - 1],
                                  alpha=1)
        self.assertTrue(np.isclose(y[0], np.float32(0.0)))
        self.assertTrue(np.isclose(y[1], np.float32(0.0)))
        self.assertTrue(np.isclose(y[2], np.float32(1.0)))


    def test_basic_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = basic_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph(in_model, keras_impl)

        self._run_model_grad_test(graph, keras_impl)

    def test_advanced_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = advenced_model(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph(in_model, keras_impl)

        self._run_model_grad_test(graph, keras_impl)

    def test_model_grad_with_output_replacements(self):
        input_shape = (8, 8, 3)
        in_model = model_with_output_replacements(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph(in_model, keras_impl)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]

        input_tensors = {inode: representative_dataset()[0] for inode in graph.get_inputs()}
        output_nodes = [graph.get_topo_sorted_nodes()[-2]]
        output_indices = [len(interest_points) - 2, len(interest_points) - 1]

        x = keras_impl.model_grad(graph_float=graph,
                                  model_input_tensors=input_tensors,
                                  interest_points=interest_points,
                                  output_list=output_nodes,
                                  all_outputs_indices=output_indices,
                                  alpha=0.3)

        # Checking that the weights where computed and normalized correctly
        self.assertTrue(np.isclose(np.sum(x), 1))

        # Checking replacement output correction
        y = keras_impl.model_grad(graph_float=graph,
                                  model_input_tensors=input_tensors,
                                  interest_points=interest_points,
                                  output_list=output_nodes,
                                  all_outputs_indices=output_indices,
                                  alpha=0)

        # Checking that the weights where computed and normalized correctly
        zero_count = len(list(filter(lambda v: v == np.float32(0), y)))
        self.assertTrue(zero_count == 2)