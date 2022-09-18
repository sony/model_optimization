import keras
import unittest

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet import ResNet50
from keras.layers import Activation, Add
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
from tensorflow import initializers
import numpy as np
import tensorflow as tf

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.keras.reader.reader import model_reader
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_tp_model, \
    get_op_quantization_configs
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc

import model_compression_toolkit as mct
tp = mct.target_platform


def simple_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    x_bn = BatchNormalization()(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def complex_model(input_shape):
    """
    This is a model that has all the different situations that define different structures for the memory graph
    which is used to run astar.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    x_bn = BatchNormalization()(x)
    x_relu = ReLU()(x_bn)
    y = tf.split(x_relu, num_or_size_splits=2, axis=0)
    x1 = Conv2D(2, 3)(y[0])
    x2 = Conv2D(2, 3)(y[1])
    concat = keras.layers.Concatenate()([x1, x2])
    x_bn2 = BatchNormalization()(concat)
    x_relu2 = Activation('relu')(x_bn2)
    outputs = Add()([x_relu2, concat])
    return keras.Model(inputs=inputs, outputs=outputs)


class TestGraphMaxCut(unittest.TestCase):

    def test_graph_max_cut_plain_graph_simple(self):
        input_shape = (8, 8, 3)
        model = simple_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        max_cut_size, _ = compute_graph_max_cut(memory_graph)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_complex(self):
        input_shape = (8, 8, 3)
        model = complex_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        max_cut_size, _ = compute_graph_max_cut(memory_graph)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_real_model(self):
        model = MobileNetV2()
        # model = ResNet50()
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        max_cut_size, schedule = compute_graph_max_cut(memory_graph, n_iter=50, astar_n_iter=500)
        self.assertIsNotNone(schedule)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)
