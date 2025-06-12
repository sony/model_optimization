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
from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.keras.default_framework_info import KerasInfo


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


class BaseTest(unittest.TestCase):
    def setUp(self):
        set_fw_info(KerasInfo)


class TestMaxCutAstarInit(BaseTest):

    def _run_max_cut_astar_initialization_test(self, model):
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        # Verify source cut creation
        self.assertTrue(mc_astar.src_cut is not None)
        self.assertTrue(len(mc_astar.src_cut.op_record) == 1)
        self.assertTrue(len(mc_astar.src_cut.mem_elements.elements) == 1)
        self.assertTrue(mc_astar.src_cut.memory_size() == 0)

        # Verify target cut creation
        self.assertTrue(mc_astar.target_cut is not None)
        self.assertTrue(len(mc_astar.target_cut.op_record) == 0)
        self.assertTrue(len(mc_astar.target_cut.mem_elements.elements) == 2)
        self.assertTrue(mc_astar.target_cut.memory_size() == 0)

    def test_max_cut_astar_init_simple(self):
        model = simple_model((8, 8, 3))
        self._run_max_cut_astar_initialization_test(model)

    def test_max_cut_astar_init_complex(self):
        model = complex_model((8, 8, 3))
        self._run_max_cut_astar_initialization_test(model)


class TestMaxCutAstarCleanMemory(BaseTest):
    def test_max_cut_astar_clean_memory_simple(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        nodes = mc_astar.src_cut.op_order
        assert len(nodes) == 1, "Only the dummy source node should be in the initial list"
        mem_elements = mc_astar.src_cut.mem_elements
        n = mc_astar.memory_graph.activation_tensor_children(mc_astar.memory_graph.operation_node_children(nodes[0])[0])[0]

        while n != graph.get_topo_sorted_nodes()[-1]:
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
        while n != graph.get_topo_sorted_nodes()[-1]:
            nodes.append(n)
            act_tensor = set(mc_astar.memory_graph.operation_node_children(n))
            mem_elements.add_elements_set(act_tensor)
            cut = Cut(nodes, set(nodes), mem_elements)
            clean_cut = mc_astar.clean_memory_for_next_step(cut)
            total_memory.append((cut.memory_size(), clean_cut.memory_size()))

            n = mc_astar.memory_graph.activation_tensor_children(list(act_tensor)[0])[0]
            mem_elements = clean_cut.mem_elements

        self.assertTrue(any([cut_size > clean_cut_size for cut_size, clean_cut_size in total_memory]))


class TestMaxCutAstarCanExpand(BaseTest):

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


class TestMaxCutAstarIsPivot(BaseTest):

    def test_max_cut_astar_is_pivot(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        nodes = mc_astar.src_cut.op_order
        assert len(nodes) == 1, "Only the dummy source node should be in the initial list"
        mem_elements = mc_astar.src_cut.mem_elements
        n = mc_astar.memory_graph.activation_tensor_children(mc_astar.memory_graph.operation_node_children(nodes[0])[0])[0]

        while n != graph.get_topo_sorted_nodes()[-1]:
            nodes.append(n)
            act_tensor = set(mc_astar.memory_graph.operation_node_children(n))
            self.assertTrue(len(list(act_tensor)) == 1)

            mem_elements.add_elements_set(act_tensor)
            cut = Cut(nodes, set(nodes), mem_elements)
            self.assertTrue(mc_astar.is_pivot(cut))

            n = mc_astar.memory_graph.activation_tensor_children(list(act_tensor)[0])[0]


class TestMaxCutAstarExpand(BaseTest):

    def test_max_cut_astar_expand_simple(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        # Expand source cut (add input node and activation tensor)
        src_expand = mc_astar.expand(mc_astar.src_cut)
        self.assertTrue(len(src_expand) == 1)
        expanded_cut = src_expand[0]
        input_act_tensor = [t.total_size for t in memory_graph.b_nodes if 'input' in t.node_name][0]
        self.assertTrue(expanded_cut.memory_size() == input_act_tensor)
        input_node = graph.get_topo_sorted_nodes()[0]
        self.assertTrue(input_node in expanded_cut.op_record)

        # Expand one more time - add conv node and activation tensor
        input_expand = mc_astar.expand(expanded_cut)
        self.assertTrue(len(input_expand) == 1)
        expanded_cut = input_expand[0]
        conv_act_tensor = [t.total_size for t in memory_graph.b_nodes if 'conv' in t.node_name][0]
        self.assertTrue(expanded_cut.memory_size() == input_act_tensor + conv_act_tensor)
        conv_node = graph.get_topo_sorted_nodes()[1]
        self.assertTrue(conv_node in expanded_cut.op_record)

    def test_max_cut_astar_expand_complex(self):
        model = complex_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        mc_astar = MaxCutAstar(memory_graph)

        # Test split node expansion
        split_node = graph.get_topo_sorted_nodes()[4]
        split_act_tensor = set([t for t in memory_graph.b_nodes if 'split' in t.node_name])
        cut = Cut([split_node], {split_node},
                  MemoryElements(split_act_tensor, sum([t.total_size for t in split_act_tensor])))

        expanded_cuts = mc_astar.expand(cut)
        self.assertTrue(len(expanded_cuts) == 2)

        conv_after_split_node1 = graph.get_topo_sorted_nodes()[5]
        conv_after_split_node2 = graph.get_topo_sorted_nodes()[6]
        self.assertTrue((conv_after_split_node1 in expanded_cuts[0].op_record and
                        conv_after_split_node2 not in expanded_cuts[0].op_record) or
                        (conv_after_split_node1 not in expanded_cuts[0].op_record and
                         conv_after_split_node2 in expanded_cuts[0].op_record))

        self.assertTrue((conv_after_split_node1 in expanded_cuts[1].op_record and
                         conv_after_split_node2 not in expanded_cuts[1].op_record) or
                        (conv_after_split_node1 not in expanded_cuts[1].op_record and
                         conv_after_split_node2 in expanded_cuts[1].op_record))

        self.assertTrue(len(expanded_cuts[0].mem_elements.elements) == 3)
        self.assertTrue(len(expanded_cuts[1].mem_elements.elements) == 3)

        # Test expansion for add node
        concat_node = graph.get_topo_sorted_nodes()[7]
        second_relu_node = graph.get_topo_sorted_nodes()[9]
        second_relu_act_tensor = mc_astar.memory_graph.operation_node_children(second_relu_node)[0]
        concat_act_tensor = mc_astar.memory_graph.operation_node_children(concat_node)[0]
        act_tensors = [second_relu_act_tensor, concat_act_tensor]
        add_node = graph.get_topo_sorted_nodes()[10]

        cut = Cut([concat_node, second_relu_node], {concat_node, second_relu_node},
                  MemoryElements(set(act_tensors), sum([t.total_size for t in act_tensors])))

        expanded_cuts = mc_astar.expand(cut)
        self.assertTrue(len(expanded_cuts) == 2)
        self.assertTrue(any([add_node in c.op_record for c in expanded_cuts]))

        self.assertTrue(len(expanded_cuts[0].mem_elements.elements) == 3)
        self.assertTrue(len(expanded_cuts[1].mem_elements.elements) == 3)


class TestMaxCutAstarSolve(BaseTest):

    def test_max_cut_astar_solve_simple(self):
        model = simple_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        l_bound = memory_graph.memory_lbound_single_op
        u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
        estimate = (u_bound + l_bound) / 2

        mc_astar = MaxCutAstar(memory_graph)

        solution = mc_astar.solve(iter_limit=10, estimate=estimate)
        self.assertIsNotNone(solution)
        path, cost, cuts = solution

        self.assertTrue(cost >= memory_graph.memory_lbound_single_op)
        for i, n in enumerate(graph.get_topo_sorted_nodes()):
            self.assertTrue(n == path[i])

    def test_max_cut_astar_solve_complex(self):
        model = complex_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        l_bound = memory_graph.memory_lbound_single_op
        u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
        estimate = (u_bound + l_bound) / 2

        mc_astar = MaxCutAstar(memory_graph)

        solution = mc_astar.solve(iter_limit=20, estimate=estimate)
        self.assertIsNotNone(solution)
        path, cost, cuts = solution

        all_tensors_sizes = [t.total_size for t in memory_graph.b_nodes]
        self.assertTrue(max(all_tensors_sizes) < cost < sum(all_tensors_sizes))
        self.assertTrue(cost >= memory_graph.memory_lbound_single_op)

        for n in graph.get_topo_sorted_nodes():
            self.assertTrue(n in path)

    def test_max_cut_astar_solve_expand(self):
        model = expanding_model((8, 8, 3))
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        l_bound = memory_graph.memory_lbound_single_op
        u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
        estimate = (u_bound + l_bound) / 2

        mc_astar = MaxCutAstar(memory_graph)

        solution = mc_astar.solve(iter_limit=20, estimate=estimate)
        self.assertIsNotNone(solution)
        path, cost, cuts = solution

        all_tensors_sizes = [t.total_size for t in memory_graph.b_nodes]
        self.assertTrue(max(all_tensors_sizes) < cost < sum(all_tensors_sizes))
        self.assertTrue(cost > memory_graph.memory_lbound_single_op)

        for n in graph.get_topo_sorted_nodes():
            self.assertTrue(n in path)


if __name__ == '__main__':
    unittest.main()
