import keras
import unittest

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Activation, Add
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
import tensorflow as tf

from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.keras.reader.reader import model_reader

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


def expanding_model(input_shape):
    """
    This is a model has a split which afterwards increases the size of the output tensor in one of the split paths.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    y = tf.split(x, num_or_size_splits=2, axis=0)
    x1 = Conv2D(2, 3)(y[0])
    x2 = Conv2D(2, 3)(y[1])
    x_expand = Conv2D(20, 1)(x2)
    x_relu = Activation('relu')(x_expand)
    x_shrink = Conv2D(2, 1)(x_relu)
    concat = keras.layers.Concatenate()([x1, x_shrink])
    return keras.Model(inputs=inputs, outputs=concat)


class TestGraphMaxCut(unittest.TestCase):

    def test_graph_max_cut_plain_graph_simple(self):
        input_shape = (8, 8, 3)
        model = simple_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_complex(self):
        input_shape = (8, 8, 3)
        model = complex_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_expanding(self):
        input_shape = (8, 8, 3)
        model = expanding_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_real_model(self):
        model = MobileNetV2()
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph, n_iter=50, astar_n_iter=500)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)
