# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from collections import namedtuple

from copy import copy, deepcopy
from typing import List

import networkx as nx
import numpy as np

from networkx.algorithms.dag import topological_sort

from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.edge import EDGE_SINK_INDEX, EDGE_SOURCE_INDEX
from model_compression_toolkit.common.graph.edge import Edge, convert_to_edge
from model_compression_toolkit.common.graph.graph_searches import GraphSearches
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.common.collectors.statistics_collector import scale_statistics, shift_statistics
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.common.logger import Logger
from model_compression_toolkit.common.hardware_representation.hardware2framework import FrameworkHardwareModel

OutTensor = namedtuple('OutTensor', 'node node_out_index')



class Graph(nx.MultiDiGraph, GraphSearches):
    """
    Base graph representing a model to be optimized.
    """

    def __init__(self,
                 name: str,
                 nodes: List[BaseNode],
                 input_nodes: List[BaseNode],
                 output_nodes: List[OutTensor],
                 edge_list: List[Edge],
                 fw_info: FrameworkInfo = None,
                 **attr):
        """
        Args:
            nodes: List of nodes the graph has.
            input_nodes: List of input nodes the model
            output_nodes: List of output nodes of the model to a list of their output indices.
            edge_list: List of edges the graph has between nodes.
            fw_info: FrameworkInfo object (needed for computing the graph's weights memory).
            **attr: Attributes to add to graph as key=value pairs.
        """

        super().__init__(**attr)
        self.name = name
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.node_to_out_stats_collector = dict()
        self.node_to_in_stats_collector = dict()
        self.add_nodes_from(nodes)
        for e in edge_list:
            self.add_edge(e.source_node,
                          e.sink_node,
                          **e.get_attributes())
        self.user_info = UserInformation()
        self.fw_info = fw_info

    def set_fw_info(self,
                   fw_info: FrameworkInfo):
        """
        Set the graph's framework info.
        Args:
            fw_info: FrameworkInfo object.
        """

        self.fw_info = fw_info

    def set_fw_hw_model(self,
                        fw_hw_model: FrameworkHardwareModel):
        """
        Set the graph's framework hardware model.
        Args:
            fw_hw_model: FrameworkHardwareModel object.
        """
        self.fw_hw_model = fw_hw_model


    def get_topo_sorted_nodes(self):
        """
        Returns: a list of toposorted nodes.
        """

        return list(nx.algorithms.dag.topological_sort(self))

    def get_op_list(self) -> np.ndarray:
        """
        Returns: Set of operators in the graph.
        """

        return np.unique([n.op for n in self.nodes()])

    def get_inputs(self) -> List[BaseNode]:
        """
        Returns: List containing the model input nodes.
        """

        return self.input_nodes

    def get_outputs(self) -> List[OutTensor]:
        """
        Returns: Dictionary containing the model output nodes.
        """

        return self.output_nodes

    def set_inputs(self,
                   input_nodes: List[BaseNode]):
        """
        Set the graph inputs dictionary.
        Args:
            input_nodes: List of the model's inputs.
        """

        self.input_nodes = input_nodes

    def set_outputs(self,
                    output_nodes: List[OutTensor]):
        """
        Set the graph outputs dictionary.
        Args:
            output_nodes: Dictionary of the model's outputs.
        """

        self.output_nodes = output_nodes

    def set_out_stats_collector_to_node(self,
                                        n: BaseNode,
                                        stats_collector: BaseStatsCollector):
        """
        Set an output statistics collector of a node in the graph, and set this statistics collector as an input
        statistics collector of nodes next to this given node.

        Args:
            n: Node to set its output statistics collector.
            stats_collector: Output statistics collector to collect the node's output statistics.
        """

        n_outputs = 1 if isinstance(n.output_shape, tuple) else len(n.output_shape)

        if n_outputs != 1:  # Node has multiple outputs
            stats_collectors = [deepcopy(stats_collector) for i in
                                range(n_outputs)]  # Create multiple tensors to attach to each next
            # node
            out_edges = self.out_edges(n, sort_by_attr=EDGE_SOURCE_INDEX)
            for sc, oe in zip(stats_collectors, out_edges):  # Attach to each next node an input tensor
                in_nodes = [e.source_node for e in self.incoming_edges(oe.sink_node,
                                                                       sort_by_attr=EDGE_SINK_INDEX)]
                if len(in_nodes) != 1:  # Next node has multiple in_nodes (besides the passed node)
                    # so self.input_tensors_dict[dst_node] should be a list of tensors
                    if oe.sink_node not in self.node_to_in_stats_collector:
                        # first visit: init tensors list as a list of Nones
                        self.node_to_in_stats_collector.update({oe.sink_node: [None] * len(in_nodes)})
                    self.node_to_in_stats_collector[oe.sink_node][in_nodes.index(n)] = sc

                else:
                    self.node_to_in_stats_collector.update({oe.sink_node: sc})
            self.node_to_out_stats_collector.update(
                {n: stats_collectors})  # Attach the passed node a list of output tensors

        else:
            self.node_to_out_stats_collector.update({n: stats_collector})
            for oe in self.out_edges(n):
                in_nodes = [e.source_node for e in self.incoming_edges(oe.sink_node,
                                                                       sort_by_attr=EDGE_SINK_INDEX)]
                if len(in_nodes) != 1:  # Next node has multiple in_nodes (besides the passed node)
                    # so self.input_tensors_dict[dst_node] should be a list of tensors
                    if oe.sink_node not in self.node_to_in_stats_collector:
                        # first visit: init tensors list as a list of Nones
                        self.node_to_in_stats_collector.update({oe.sink_node: [None] * len(in_nodes)})
                    self.node_to_in_stats_collector[oe.sink_node][in_nodes.index(n)] = stats_collector

                else:
                    self.node_to_in_stats_collector.update({oe.sink_node: stats_collector})

    def get_out_stats_collector(self,
                                n: BaseNode) -> BaseStatsCollector:
        """
        Get the output statistics collector of a node containing output statistics of the node.
        Args:
            n: Node to get its output statistics collector.

        Returns:
            Tensor containing output statistics of the node.
        """
        return self.node_to_out_stats_collector.get(n)

    def get_in_stats_collector(self,
                               n: BaseNode) -> BaseStatsCollector:
        """
        Get the input statistics collector of a node containing input statistics of the node.
        Args:
            n: Node to get its input statistics collector.

        Returns:
            Statistics collector containing input statistics of the node.
        """

        sc = self.node_to_in_stats_collector.get(n)
        if sc is None:
            raise Exception()
        return sc

    def scale_stats_collector(self,
                              node: BaseNode,
                              scale_factor: np.ndarray):
        """
        Scale the output statistics of a node in the graph by a given scaling factor.
        The scaling factor can be a single value (scale per-tensor) or multiple values (scale per-channel).
        This is done in place.

        Args:
            node: Node to scale its output statistics.
            scale_factor: Scale factor to use for the statistics scaling.

        """

        sc = self.get_out_stats_collector(node)
        scaled_sc = scale_statistics(sc, scale_factor)
        self.set_out_stats_collector_to_node(node, scaled_sc)

    def shift_stats_collector(self,
                              node: BaseNode,
                              shift_value: np.ndarray):
        """
        Shift the output statistics of a node in the graph by a given value.
        The shifting value can be a single value (shifting per-tensor) or multiple values (shifting per-channel).
        This is done in place.

        Args:
            node: Node to scale its output statistics.
            shift_value: Value to use for the statistics shifting.

        """

        sc = self.get_out_stats_collector(node)
        shifted_sc = shift_statistics(sc, shift_value)
        self.set_out_stats_collector_to_node(node, shifted_sc)

    def find_node_by_name(self,
                          name: str) -> List[BaseNode]:
        """
        Find and return a list of nodes by a name.

        Args:
            name: Name to find nodes in the graph by.

        Returns:
            List of nodes named
        """

        return [n for n in self.nodes if n.name == name]

    def get_next_nodes(self,
                       node_obj: BaseNode) -> List[BaseNode]:
        """
        Get next nodes (in a topological order) of a node.

        Args:
            node_obj: Node to get its next nodes.

        Returns:
            List of output nodes objects.

        """

        return [edges_list.sink_node for edges_list in self.out_edges(node_obj)]

    def get_prev_nodes(self,
                       node_obj: BaseNode) -> List[BaseNode]:
        """
        Get previous nodes (in a topological order) of a node.

        Args:
            node_obj: Node to get its previous nodes.

        Returns:
            List of input nodes objects.

        """

        return [edges_list.source_node for edges_list in self.incoming_edges(node_obj)]

    def reconnect_out_edges(self,
                            current_node: BaseNode,
                            new_node: BaseNode):
        """
        Connect all outgoing edges of a node to be outgoing edges of a different node
        (useful when replacing a node during substitutions).

        Args:
            current_node: Node to reconnect its outgoing edges.
            new_node: Node to connect the outgoing edges of the current node to.
        """

        for oe in self.out_edges(current_node):
            self.add_edge(new_node, oe.sink_node, **oe.get_attributes())
            self.remove_edge(current_node, oe.sink_node)

    def reconnect_in_edges(self,
                           current_node: BaseNode,
                           new_node: BaseNode):
        """
        Connect all incoming edges of a node to be incoming edges of a different node
        (useful when replacing a node during substitutions).

        Args:
            current_node: Node to reconnect its incoming edges.
            new_node: Node to connect the incoming edges of the current node to.
        """

        for ie in self.incoming_edges(current_node):
            self.add_edge(ie.source_node, new_node, **ie.get_attributes())
            self.remove_edge(ie.source_node, current_node)

    def add_node_with_in_edges(self, new_node: BaseNode, input_nodes: List[BaseNode],
                               input_nodes_output_index: List[int] = []):
        """
        Add node to graph and connect it to its input nodes
        (useful when adding a node during substitutions).

        Args:
            new_node: Node to add.
            input_nodes: A list of new_node input nodes. The order is the sink_index of the edge
             between the input node and new_node
            input_nodes_output_index: A list output indices from input nodes. The order is the
             source_index of the edge. Deafult is an empty list which means all output indices
             are zero
        """

        if len(input_nodes_output_index) == 0:
            input_nodes_output_index = [0] * len(input_nodes)

        if len(input_nodes_output_index) != len(input_nodes):
            raise Exception('Graph.add_node_with_in_edges: input_nodes & input_nodes_output_index must be the same length')

        self.add_node(new_node)
        for sink_index, (in_node, source_index) in enumerate(zip(input_nodes, input_nodes_output_index)):
            self.add_edge(in_node, new_node, source_index=source_index, sink_index=sink_index)

    def replace_output_node(self,
                            current_node: BaseNode,
                            new_node: BaseNode):
        """
        If a node is being substituted with another node and it is an output node, the graph's outputs
        should be updated as well. This function takes care of it by going over the graph's outputs, and
        replacing the current output node with a new output node.
        If current node is not an output node, nothing gets changed.

        Args:
            current_node: Node that (possibly) is an output node.
            new_node: New node to set as an output node if the current node is an output node.

        """

        graph_outputs = self.get_outputs()
        new_graph_outputs = copy(graph_outputs)
        for graph_ot_index, ot in enumerate(graph_outputs):
            if current_node == ot.node:
                new_graph_outputs[graph_ot_index] = OutTensor(new_node, ot.node_out_index)
        self.set_outputs(new_graph_outputs)

    def remove_node(self,
                    node_to_remove: BaseNode,
                    new_graph_inputs: List[BaseNode] = None,
                    new_graph_outputs: List[OutTensor] = None):
        """
        Remove a node from the graph. A new inputs/outputs lists can be passed in case the node is currently an
        input/output of the graph. If the node is an input/output and a new list wasn't passed an error is logged.

        Args:
            node_to_remove: Node to remove from the graph.
            new_graph_inputs: An inputs list to set as the graph's inputs.
            new_graph_outputs: An outputs list to set as the graph's outputs.

        """

        output_nodes = [ot.node for ot in self.get_outputs()]  # get output nodes from namedtuples
        if node_to_remove in output_nodes:  # If node is in the graph's outputs, the outputs should be updated
            if new_graph_outputs is None:
                Logger.critical(f'{node_to_remove.name} is in graph outputs, but new outputs were not given.')
            self.set_outputs(new_graph_outputs)

        if node_to_remove in self.get_inputs():  # If node is in the graph's inputs, the inputs should be updated
            if new_graph_inputs is None:
                Logger.critical(f'{node_to_remove.name} is in graph inputs, but new inputs were not given.')
            self.set_inputs(new_graph_inputs)

        # Make sure there are no connected edges left to the node before removing it.
        assert len(
            self.incoming_edges(node_to_remove)) == 0, f'There are {len(self.incoming_edges(node_to_remove))} ' \
                                                       f'incoming ' \
                                                       f'edges to node {node_to_remove}, and they should be removed ' \
                                                       f'before deleting the node from the graph.'
        assert len(self.out_edges(node_to_remove)) == 0, f'There are {len(self.out_edges(node_to_remove))} outgoing ' \
                                                         f'edges to node {node_to_remove}, and they should be removed ' \
                                                         f'' \
                                                         f'' \
                                                         f'before deleting the node from the graph.'
        #  Remove node
        super().remove_node(node_to_remove)

    def incoming_edges(self,
                       n: BaseNode,
                       sort_by_attr: str = None) -> List[Edge]:
        """
        Get a list of incoming edges of a node. If sort_by_attr is passed, the returned list
        is sorted by that edge's attribute.

        Args:
            n: Node to get its incoming edges.
            sort_by_attr: Attribute to sort the edges by.

        Returns:
            List of incoming edges of the node. Each incoming edge is a tuple of:
            (source node, destination node, edge data)
        """

        input_edges = [convert_to_edge(e) for e in super().in_edges(n, data=True)]
        if sort_by_attr is not None:
            input_edges.sort(key=lambda e: getattr(e, sort_by_attr))
        return input_edges

    def out_edges(self,
                  n: BaseNode,
                  sort_by_attr: str = None) -> List[Edge]:
        """
        Get a list of outgoing edges of a node. If sort_by_attr is passed, the returned list
        is sorted by that edge's attribute.

        Args:
            n: Node to get its outgoing edges.
            sort_by_attr: Attribute to sort the edges by.

        Returns:
            List of outgoing edges of the node.
        """

        output_edges = [convert_to_edge(e) for e in super().edges(n, data=True)]
        if sort_by_attr is not None:
            output_edges.sort(key=lambda e: getattr(e, sort_by_attr))
        return output_edges

    def get_memory(self) -> float:
        """

        Returns: Total memory consumption of the graph in bytes.

        """
        memory = 0
        for n in self.nodes:
            memory += n.get_memory_bytes(self.fw_info)
        return memory

    def get_configurable_sorted_nodes_names(self,
                                            include_reused_nodes: bool = False) -> List[str]:
        """
        Get a list of nodes' names that can be configured (namely, has one or
        more weight qc candidate). The names are sorted according to the topological
        order of the graph.

        Args:
            include_reused_nodes: Whether or not to include reused nodes (False by default).

        Returns: List of nodes' names that can be configured (namely, has one or
        more weight qc candidate) sorted topology.

        """
        sorted_names = [n.name for n in self.get_configurable_sorted_nodes(include_reused_nodes=include_reused_nodes)]
        return sorted_names

    def get_configurable_sorted_nodes(self,
                                      include_reused_nodes: bool = False) -> List[BaseNode]:
        """
        Get a list of nodes that can be configured (namely, has one or
        more weight qc candidate and their weights should be quantized).
        The nodes are sorted according to the topological order of the graph.

        Args:
            include_reused_nodes: Whether or not to include reused nodes (False by default).

        Returns:
            A list of nodes that can be configured (namely, has one or more weight qc candidate) sorted topology.

        """
        sorted_configurable_nodes = []
        sorted_nodes = list(topological_sort(self))
        for n in sorted_nodes:
            if n.is_weights_quantization_enabled():
                if not n.reuse or include_reused_nodes:
                    if len(n.candidates_quantization_cfg) >= 1:
                        sorted_configurable_nodes.append(n)
        return sorted_configurable_nodes
