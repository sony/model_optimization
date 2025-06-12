# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from keras import Input
from keras.layers import Conv2D, BatchNormalization, ReLU, Activation, Add
import tensorflow as tf

from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.keras.reader.reader import model_reader
from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.keras.default_framework_info import KerasInfo


def basic_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    x_bn = BatchNormalization()(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def node_multiple_outputs_model(input_shape):
    inputs = Input(shape=input_shape)
    y = tf.split(inputs, num_or_size_splits=2, axis=0)
    x1 = Conv2D(2, 3)(y[0])
    x2 = Conv2D(2, 3)(y[1])
    outputs = keras.layers.Concatenate()([x1, x2])
    return keras.Model(inputs=inputs, outputs=outputs)


def residual_model(input_shape):
    inputs = Input(shape=input_shape)
    y = Conv2D(7, 8)(inputs)
    x = BatchNormalization()(y)
    x = Activation('relu')(x)
    outputs = Add()([x, y])
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def _is_bipartite(g):
    queue = [list(g.nodes)[0]]
    side_a = {list(g.nodes)[0]}
    side_b = set()

    while len(queue) > 0:
        u = queue.pop()
        u_side = 0 if u in side_a else 1
        for e in g.out_edges(u):
            v = e[1]
            if v not in side_a and v not in side_b:
                if u_side == 0:
                    side_b.add(v)
                else:
                    side_a.add(v)
            else:
                v_side = 0 if v in side_a else 1
                if v_side == u_side:
                    return False
    return True


class TestMemoryGraph(unittest.TestCase):
    def setUp(self):
        set_fw_info(KerasInfo)

    def test_memory_graph_build(self):
        model = basic_model((8, 8, 3))

        graph = model_reader(model)

        memory_graph = MemoryGraph(graph)

        self.assertTrue(len(memory_graph.a_nodes) == 4)
        self.assertTrue(len(memory_graph.b_nodes) == 4)
        self.assertTrue(graph.get_topo_sorted_nodes()[0].name in [node.name for node in memory_graph.sources_a])
        self.assertTrue(len(memory_graph.sinks_b) == 1)
        self.assertTrue(memory_graph.memory_lbound_single_op == 264)

        self.assertTrue(_is_bipartite(memory_graph))

    def test_memory_graph_node_with_multiple_outputs(self):
        model = node_multiple_outputs_model((8, 8, 3))

        graph = model_reader(model)

        memory_graph = MemoryGraph(graph)

        self.assertTrue(len(memory_graph.a_nodes) == 5)
        self.assertTrue(len(memory_graph.b_nodes) == 6)
        self.assertTrue(graph.get_topo_sorted_nodes()[0].name in [node.name for node in memory_graph.sources_a])
        self.assertTrue(len(memory_graph.sinks_b) == 1)
        self.assertTrue(memory_graph.memory_lbound_single_op == 576)

        self.assertTrue(_is_bipartite(memory_graph))

        # Check that the convolution node that goes to two different
        self.assertTrue(len(set([t.node_output_index for t in memory_graph.b_nodes if 'split' in t.node_name])) == 2)

    def test_memory_graph_with_residual(self):
        model = residual_model((8, 8, 3))

        graph = model_reader(model)

        memory_graph = MemoryGraph(graph)

        self.assertTrue(len(memory_graph.a_nodes) == 5)
        self.assertTrue(len(memory_graph.b_nodes) == 5)
        self.assertTrue(graph.get_topo_sorted_nodes()[0].name in [node.name for node in memory_graph.sources_a])
        self.assertTrue(len(memory_graph.sinks_b) == 1)
        self.assertTrue(memory_graph.memory_lbound_single_op == 199)

        self.assertTrue(_is_bipartite(memory_graph))


if __name__ == '__main__':
    unittest.main()
