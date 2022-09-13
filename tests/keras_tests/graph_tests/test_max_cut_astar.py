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

from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.max_cut_astar import MaxCutAstar
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.keras.reader.reader import model_reader


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


class TestMaxCutAstarInit(unittest.TestCase):

    def _run_max_cut_astar_initialization_test(self, model):
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        # Verify source cut creation
        self.assertTrue(mc_astar.src_cut is not None)
        self.assertTrue(len(mc_astar.src_cut.op_record) == 1)
        self.assertTrue(len(mc_astar.src_cut.mem_elements.elements) == 1)
        self.assertTrue(mc_astar.src_cut.mem_elements.total_size == 0)

        # Verify target cut creation
        self.assertTrue(mc_astar.target_cut is not None)
        self.assertTrue(len(mc_astar.target_cut.op_record) == 0)
        self.assertTrue(len(mc_astar.target_cut.mem_elements.elements) == 2)
        self.assertTrue(mc_astar.target_cut.mem_elements.total_size == 0)

    def test_max_cut_astar_init_simple(self):
        model = simple_model((8, 8, 3))
        self._run_max_cut_astar_initialization_test(model)

    def test_max_cut_astar_init_complex(self):
        model = complex_model((8, 8, 3))
        self._run_max_cut_astar_initialization_test(model)


class TestMaxCutAstarCleanMemory(unittest.TestCase):

    def test_max_cut_astar_clean_memory_simple(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        nodes = mc_astar.src_cut.op_order
        assert len(nodes) == 1, "Only the dummy source node should be in the initial list"
        mem_elements = mc_astar.src_cut.mem_elements
        n = mc_astar.memory_graph.activation_tensor_children(mc_astar.memory_graph.operation_node_children(nodes[0])[0])[0]

        while n not in mc_astar.memory_graph.sinks_a:
            nodes.append(n)
            act_tensor = set(mc_astar.memory_graph.operation_node_children(n))
            self.assertTrue(len(list(act_tensor)) == 1)

            mem_elements.add_elements_set(act_tensor)
            cut = Cut(nodes, set(nodes), mem_elements)
            self.assertTrue(len(cut.mem_elements.elements) == 2)

            clean_cut = mc_astar.clean_memory_for_next_step(cut)
            self.assertTrue(len(clean_cut.mem_elements.elements) == 1)

            n = mc_astar.memory_graph.activation_tensor_children(list(act_tensor)[0])[0]
            mem_elements = clean_cut.mem_elements

    def test_max_cut_astar_clean_memory_complex(self):
        model = complex_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        nodes = mc_astar.src_cut.op_order
        assert len(nodes) == 1, "Only the dummy source node should be in the initial list"
        mem_elements = mc_astar.src_cut.mem_elements
        n = mc_astar.memory_graph.activation_tensor_children(mc_astar.memory_graph.operation_node_children(nodes[0])[0])[0]

        # Only testing consistent and successful run, not testing any calculation correctness
        # except that at least at one point the cut got cleaner
        total_memory = []
        while n not in mc_astar.memory_graph.sinks_a:
            nodes.append(n)
            act_tensor = set(mc_astar.memory_graph.operation_node_children(n))
            mem_elements.add_elements_set(act_tensor)
            cut = Cut(nodes, set(nodes), mem_elements)
            clean_cut = mc_astar.clean_memory_for_next_step(cut)
            total_memory.append((cut.mem_elements.total_size, clean_cut.mem_elements.total_size))

            n = mc_astar.memory_graph.activation_tensor_children(list(act_tensor)[0])[0]
            mem_elements = clean_cut.mem_elements

        self.assertTrue(any([cut_size > clean_cut_size for cut_size, clean_cut_size in total_memory]))


class TestMaxCutAstarCanExpand(unittest.TestCase):

    def test_max_cut_astar_can_expand_simple(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        cut1 = Cut([], set(), MemoryElements(set(), 0))
        self.assertFalse(mc_astar.can_expand(graph.get_topo_sorted_nodes()[0], cut1),
                         "Shouldn't be able to expand a cut without memory elements")
        self.assertFalse(mc_astar.can_expand(graph.get_topo_sorted_nodes()[1], cut1),
                         "Shouldn't be able to expand a cut without memory elements")

        node0 = graph.get_topo_sorted_nodes()[0]
        node1 = graph.get_topo_sorted_nodes()[1]
        act_tensor0 = mc_astar.memory_graph.operation_node_parents(node0)[0]
        act_tensor1 = mc_astar.memory_graph.operation_node_parents(node1)[0]

        cut2 = Cut([node1], {node1}, MemoryElements({act_tensor1}, act_tensor1.total_size))
        self.assertFalse(mc_astar.can_expand(node1, cut2),
                         "Shouldn't be able to expand a cut with a node that is already in the cut")
        cut3 = Cut([], set(), MemoryElements({act_tensor0}, act_tensor0.total_size))
        self.assertFalse(mc_astar.can_expand(node1, cut3),
                         "Shouldn't be able to expand a cut with a node that misses its required memory elements "
                         "for computation")

        cut4 = Cut([node0], {node0}, MemoryElements({act_tensor1}, act_tensor1.total_size))
        self.assertTrue(mc_astar.can_expand(node1, cut4),
                         "Should be able to expand a the given cut with the given node")

    def test_max_cut_astar_can_expand_complex(self):
        model = complex_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        # Test concatenate node expansion
        conv_after_split_node1 = graph.get_topo_sorted_nodes()[5]
        conv_after_split_node2 = graph.get_topo_sorted_nodes()[6]
        conv1_act_tensor = mc_astar.memory_graph.operation_node_children(conv_after_split_node1)[0]
        conv2_act_tensor = mc_astar.memory_graph.operation_node_children(conv_after_split_node2)[0]
        cut_cant_expand = Cut([conv_after_split_node1], {conv_after_split_node1},
                              MemoryElements({conv1_act_tensor}, conv1_act_tensor.total_size))
        concat_node = graph.get_topo_sorted_nodes()[7]
        self.assertFalse(mc_astar.can_expand(concat_node, cut_cant_expand),
                         "Shouldn't be able to expand a cut by a node without all its required memory elements")

        cut_expand = Cut([conv_after_split_node1, conv_after_split_node2],
                         {conv_after_split_node1, conv_after_split_node2},
                         MemoryElements({conv1_act_tensor, conv2_act_tensor},
                                        sum([conv1_act_tensor.total_size, conv2_act_tensor.total_size])))
        self.assertTrue(mc_astar.can_expand(concat_node, cut_expand),
                        "Should be able to expand the given cut with the given node")

        # Test add node expansion
        second_relu_node = graph.get_topo_sorted_nodes()[9]
        second_relu_act_tensor = mc_astar.memory_graph.operation_node_children(second_relu_node)[0]
        concat_act_tensor = mc_astar.memory_graph.operation_node_children(concat_node)[0]
        cut_cant_expand2 = Cut([conv_after_split_node1, conv_after_split_node2, concat_node],
                               {conv_after_split_node1, conv_after_split_node2, concat_node},
                               MemoryElements({concat_act_tensor}, concat_act_tensor.total_size))
        add_node = graph.get_topo_sorted_nodes()[10]
        self.assertFalse(mc_astar.can_expand(add_node, cut_cant_expand2),
                         "Shouldn't be able to expand a cut by a node without all its required memory elements")

        cut_expand2 = Cut([concat_node, second_relu_node],
                          {concat_node, second_relu_node},
                          MemoryElements({concat_act_tensor, second_relu_act_tensor},
                                         sum([concat_act_tensor.total_size, second_relu_act_tensor.total_size])))
        self.assertTrue(mc_astar.can_expand(add_node, cut_expand2),
                        "Should be able to expand the given cut with the given node")


class TestMaxCutAstarIsPivot(unittest.TestCase):

    def test_max_cut_astar_is_pivot(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        nodes = mc_astar.src_cut.op_order
        assert len(nodes) == 1, "Only the dummy source node should be in the initial list"
        mem_elements = mc_astar.src_cut.mem_elements
        n = mc_astar.memory_graph.activation_tensor_children(mc_astar.memory_graph.operation_node_children(nodes[0])[0])[0]

        while n not in mc_astar.memory_graph.sinks_a:
            nodes.append(n)
            act_tensor = set(mc_astar.memory_graph.operation_node_children(n))
            self.assertTrue(len(list(act_tensor)) == 1)

            mem_elements.add_elements_set(act_tensor)
            cut = Cut(nodes, set(nodes), mem_elements)
            self.assertTrue(mc_astar.is_pivot(cut))

            n = mc_astar.memory_graph.activation_tensor_children(list(act_tensor)[0])[0]


if __name__ == '__main__':
    unittest.main()
