import keras
import unittest
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
from tensorflow import initializers
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.back2framework.model_gradients import model_grad
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tp_model import generate_tp_model, \
    get_op_quantization_configs
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tpc_keras import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model

import model_compression_toolkit as mct
tp = mct.target_platform


def create_model_1(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def representative_dataset():
    return [np.random.randn(1, 8, 8, 3)]


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

    def test_debug_model_grad(self):
        input_shape = (8, 8, 3)
        in_model = create_model_1(input_shape)
        keras_impl = KerasImplementation()
        graph = prepare_graph(in_model, keras_impl)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [sorted_graph_nodes[1], sorted_graph_nodes[-1]]

        # model = keras_impl.model_builder(graph, ModelBuilderMode.FLOAT, sorted_graph_nodes)[0]
        # model = keras_impl.model_builder(graph, ModelBuilderMode.FLOAT, interest_points)[0]
        # inputs_list = representative_dataset()
        # tensor_data = keras_impl.run_model_inference(model, inputs_list)

        # input_tensors = {inode: tensor_data[i].numpy() for i, inode in enumerate(interest_points)}
        input_tensors = {inode: representative_dataset() for inode in graph.get_inputs()}
        x = model_grad(graph, input_tensors, interest_points, [o.node for o in graph.output_nodes])
