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


from typing import List, Dict, Tuple

from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.edge import Edge, convert_to_edge
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.reader.connectivity_handler import OutTensor


def merge_models_edges(inner_model_node: BaseNode,
                       outer_graph: Graph,
                       inner_graph: Graph) -> List[Edge]:
    """
    Given two MultiDiGraphs (one of an outer model and the second of the inner model), merge their edges into
    a single edges list representing the edges that should be in a single graph after unrolling the inner graph.
    This is done by taking in/out edges the model node has, out edges the input nodes of the inner model have, and
    rewiring them into edges after removing the inner model node from the outer model graph.
    Args:
        outer_graph: MultiDiGraph of the outer graph.
        inner_graph: MultiDiGraph of the inner graph.
        inner_model_node: Node of the inner model in the graph of the outer model.

    Returns:
        List of edges that should be in a single graph after unrolling the inner graph.
    """

    # compute in/out edges the model node has
    # edge : (source node, dst node, dict_data)
    outer_graph_edges = [convert_to_edge(e) for e in outer_graph.edges(data=True)]
    model_node_in_edges, model_node_out_edges = get_model_node_edges(inner_model_node,
                                                                     outer_graph_edges)

    # In order to know how te rewire edges that go to the model node, and the model's layers that connected to the
    # model inputs (their successors) we create a dictionary from the inner model inputs to their out edges.
    inner_inputs_out_edges = get_inner_inputs_successors(inner_graph)

    # First, conclude all edges (inner and outer models) in the final edges list. Then, rewire incoming and
    # outgoing edges of the inner model, and remove unwanted edges.
    res_edge_list = outer_graph_edges
    res_edge_list.extend([convert_to_edge(e) for e in inner_graph.edges(data=True)])
    rewire_incoming_edges(inner_inputs_out_edges,
                          inner_graph.get_inputs(),
                          model_node_in_edges,
                          res_edge_list)
    rewire_outgoing_edged(inner_graph.get_outputs(),
                          model_node_out_edges,
                          res_edge_list)

    return res_edge_list


def rewire_outgoing_edged(inner_model_outputs: List[OutTensor],
                          model_node_out_edges: List[Edge],
                          res_edges: List[Edge]):
    """
    Rewiring outgoing edges the inner model has to the inner model output nodes. This is
    Done in place (on res_edges).

    Args:
        inner_model_outputs: Inner model outputs dictionary (output node to a list of output indices).
        model_node_out_edges: Inner model outgoing edges.
        res_edges: Final edges list the outer graph should have.

    """
    for idx, out_node in enumerate(inner_model_outputs):  # iterate output nodes (they are the new source nodes of
        # new edges)
        for model_node_out_edge in model_node_out_edges:  # iterate model outgoing edges
            if model_node_out_edge.source_index == idx:  # found en edge to reconnect

                new_edge = Edge(out_node.node,  # create the new rewired edge
                                model_node_out_edge.sink_node,
                                out_node.node_out_index,
                                model_node_out_edge.sink_index)

                res_edges.append(new_edge)
                res_edges.remove(model_node_out_edge)


def rewire_incoming_edges(inner_inputs_out_edges: Dict[BaseNode, List[Edge]],
                          inner_model_inputs_dict: List[BaseNode],
                          model_node_in_edges: List[Edge],
                          res_edge_list: List[Edge]):
    """
    Rewiring incoming edges the inner model has to the inner model inputs successors. This is
    Done in place (on res_edge_list).

    Args:
        inner_inputs_out_edges: Dictionary of inner model input nodes to their out edges.
        inner_model_inputs_dict: Inner model inputs dictionary (input node to its index).
        model_node_in_edges: Inner model incoming edges.
        res_edge_list: Final edges list the outer graph should have.

    """
    for model_node_in_edge in model_node_in_edges:  # iterate incoming edges of the model node
        index = model_node_in_edge.sink_index
        for input_index, input_layer in enumerate(inner_model_inputs_dict):
            if index == input_index:  # Found an edge (or edges) to rewire
                edges_to_reconnect = inner_inputs_out_edges.get(
                    input_layer)  # the model incoming edge should be rewired to the input node successors.
                if edges_to_reconnect is not None:  # Make sure input layer is actually connected to some next layers
                    for input_out_edge in edges_to_reconnect:  # iterate edges and reconnect them to model node incoming
                        # edge in outer loop.
                        res_edge_list.append(

                            Edge(model_node_in_edge.source_node,
                                 input_out_edge.sink_node,
                                 model_node_in_edge.source_index,
                                 input_out_edge.sink_index))

                        res_edge_list.remove(input_out_edge)
        res_edge_list.remove(model_node_in_edge)


def get_inner_inputs_successors(inner_graph: Graph) -> Dict[BaseNode, List[Edge]]:
    """
    Compute out edges the input nodes of the inner model has.
    Args:
        inner_graph: ModelGraphInfo of the inner model.

    Returns:
        Dictionary of inner model input nodes to their out edges.
    """
    inner_inputs_out_edges = dict()
    inner_graph_edges = [convert_to_edge(e) for e in inner_graph.edges(data=True)]
    for e in inner_graph_edges:
        if e.source_node in inner_graph.get_inputs():
            input_node = e.source_node
            if input_node in inner_inputs_out_edges:
                inner_inputs_out_edges[input_node].append(e)
            else:
                inner_inputs_out_edges[input_node] = [e]
    return inner_inputs_out_edges


def get_model_node_edges(model_node: BaseNode,
                         outer_edge_list: List[Edge]) -> Tuple[List[Edge], List[Edge]]:
    """
    Get incoming and outgoing edges the inner model node has in the outer graph.
    Args:
        model_node: Inner model node in outer graph.
        outer_edge_list: Outer graph's edges list.

    Returns:
        Tuple of incoming edges list and outgoing edges list the inner model node has.
    """

    model_node_in_edges = []
    model_node_out_edges = []
    for e in outer_edge_list:
        if e.sink_node == model_node:
            model_node_in_edges.append(e)
        if e.source_node == model_node:
            model_node_out_edges.append(e)
    return model_node_in_edges, model_node_out_edges
