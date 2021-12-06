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


from copy import copy

import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.util.object_identity import Reference as TFReference
from typing import List

from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.edge import Edge
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.reader.common import is_node_a_model, is_node_an_input_layer
from model_compression_toolkit.keras.reader.connectivity_handler import ConnectivityHandler
from model_compression_toolkit.keras.reader.nested_model.nested_model_handler import merge_graphs

keras = tf.keras
layers = keras.layers


def get_edges(connectivity_handler: ConnectivityHandler) -> List[Edge]:
    """
    Creates a list of edges the model has. Each edge contains source and destination nodes, and indices that are
    the relative position of the tensor that connects those two nodes (in output tensors of the source node, and
    in input tensors of the destination node).
    Args:
        connectivity_handler: Connectivity handler containing all information about connections in graph.

    Returns:
        List of edges in the graph.
    """
    edges = []
    for src_node in connectivity_handler.get_nodes():  # go over all nodes in the graph
        for dst_node, src_index, dst_index in connectivity_handler.get_out_edges_params_list(
                src_node):  # get parameters of all out edges for a node
            edge = Edge(src_node,
                        dst_node,
                        src_index,
                        dst_index)  # create edge
            edges.append(edge)  # append edge to list
    return edges


def build_tensors_list(tensors_list) -> List[TFReference]:
    """
    Build list of tensor references from a list of tensors or a single tensor.
    Args:
        tensors_list: A tensor or a list of tensors.

    Returns:
        List of tensors' references.
    """
    return [tensors_list.ref()] if not isinstance(tensors_list, (list, tuple)) else [tensor.ref()
                                                                                     for
                                                                                     tensor in
                                                                                     tensors_list]

def build_connectivity_handler(model: Model) -> ConnectivityHandler:
    """
    Build a connectivity handler containing all information about connections in the model (nodes and
    interconnecting tensors).
    For a reused layer, several nodes may be generated, and can be distinguished by their names.
    Args:
        model: Keras model to build its connectivity handler.

    Returns:
        Connectivity handler containing all information about connections in graph.
    """
    connectivity_handler = ConnectivityHandler()
    for nodes in model._nodes_by_depth.values():
        for node in nodes:  # nodes_by depth values are lists (each list for a different depth)
            node_inputs = node.input_tensors if is_node_an_input_layer(node) else node.keras_inputs
            input_tensors = build_tensors_list(node_inputs)  # build input tensors of the node
            output_tensors = build_tensors_list(node.output_tensors)  # build output tensors of the node
            connectivity_handler.add_node(node,
                                          input_tensors,
                                          output_tensors)  # add the node and its connecting tensors to the
    return connectivity_handler


def build_graph(model: Model,
                connectivity_handler: ConnectivityHandler) -> Graph:
    """
    Given a Keras model, build and return an networkx MultiDiGraph containing all data (nodes, edges,
    inputs and outputs) representing that model.

    Args:
        model: Keras model to build its graph.
        connectivity_handler: Connectivity handler containing all information about connections in graph.

    Returns:
        Networkx MultiDiGraph representing the Keras model.

    """
    # connectivity handler contains info with keras nodes, but we build the model's graph with internal nodes so we
    # need to convert connectivity handler's data to internal nodes.
    connectivity_handler.convert_to_internal_nodes()
    inputs = connectivity_handler.build_inputs_list([t.ref() for t in model.inputs])
    outputs = connectivity_handler.build_outputs_list([t.ref() for t in model.outputs])
    edges = get_edges(connectivity_handler)
    nodes = connectivity_handler.get_nodes()
    return Graph(nodes, inputs, outputs, edges)


def parse_model(model: Model) -> Graph:
    """
    Parse a Keras model into a Graph.
    In case of a nested model, it recursively unrolls inner models.

    Args:
        model: Keras model to build its graph.

    Returns:
        Networkx MultiDiGraph representing the Keras model including: nodes, edges, inputs, and outputs.
    """
    connectivity_handler = build_connectivity_handler(model)
    model_graph = build_graph(model, connectivity_handler)

    # Go over all nodes in the graph, and if one of them is a model by itself, unroll it recursively by
    # merging the inner model's graph into the outer model's graph.
    nodes = copy(model_graph.nodes)
    for node in nodes:
        if is_node_a_model(node):  # if the node represents a Keras model - flat it recursively
            model_graph = flatten_nested_model(model_graph,
                                               node,
                                               model)
    return model_graph


def flatten_nested_model(outer_graph: Graph,
                         inner_model_node: BaseNode,
                         outer_keras_model: Model):
    """
    Flat a nested model given two graphs: inner and outer models' graphs.
    The resulting graph contains all nodes of the inner model (except for input nodes and the node representing the
    model itself). Also, outputs are updated if the inner model is an output of the outer model (by inner model's
    output nodes), and edges to the inner model node are rewired.

    Args:
        outer_graph: Networkx MultiDiGraph of outer model.
        inner_model_node: Node in outer_graph representing the inner model.
        outer_keras_model: Keras model for retrieving the inner Keras model.

    Returns:
        Networkx MultiDiGraph object representing the Keras model when inner model is flattened.
    """

    inner_model = outer_keras_model.get_layer(inner_model_node.name)  # get the Keras model
    inner_graph = parse_model(inner_model)  # recursively parse this model into graph.
    # merge graphs into a single graph representing the model when the inner model is being unrolled.
    outer_flatten_graph = merge_graphs(inner_model_node,
                                       outer_graph,
                                       inner_graph)
    return outer_flatten_graph


def model_reader(keras_model: Model) -> Graph:
    """
    Reads a Keras model and builds a base graph representing the model.
    Args:
        keras_model: Keras model to build its graph representation.

    Returns:
        Base graph of the Keras model.
    """
    logging.info("Start Model Reading...")
    graph = parse_model(keras_model)
    return graph
