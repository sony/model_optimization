# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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


import logging
from typing import Callable, Dict

import numpy as np
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.pytorch.reader.graph_builders import edges_builder, nodes_builder
from model_compression_toolkit.core.pytorch.utils import set_model


def generate_module_dict(model: torch.nn.Module) -> Dict:
    """
    Creates a dictionary from the Pytorch model's named modules by module name.

    Args:
        model: A Pytorch model.

    Returns:
        A dictionary of the Pyotrch model's named modules.
    """
    module_dict = dict()
    for name, module in model.named_modules():
        module_dict[name] = module
    return module_dict


def build_graph(model: torch.fx.GraphModule,
                to_numpy: Callable) -> Graph:
    """
    Given a Pytorch FX model, build and return an networkx MultiDiGraph containing all data (nodes, edges,
    inputs and outputs) representing that model.

    Args:
        model: Pytorch FX model to build its graph.
        to_numpy: Function to convert framework's tensor to a Numpy array.

    Returns:
        Networkx MultiDiGraph representing the Pytorch model.

    """
    # generate a dictionary with all model modules
    module_dict = generate_module_dict(model)

    # convert fx nodes to generic graph nodes
    nodes, inputs, outputs, fx_node_2_graph_node = nodes_builder(model, module_dict, to_numpy)

    # build graph edges
    edges = edges_builder(model, fx_node_2_graph_node)

    return Graph(model._get_name(), nodes, inputs, outputs, edges)


def fx_graph_module_generation(pytorch_model: torch.nn.Module,
                               representative_data_gen: Callable,
                               to_tensor: Callable) -> torch.fx.GraphModule:
    """
    Generates a fx.GraphModule from a torch.nn.Module.

    Args:
        pytorch_model: A dynamic Pytorch model.
        representative_data_gen (Callable): Representative dataset used for shape inference.
        to_tensor: Function to convert a Numpy array to a framework's tensor.

    Returns:
        A fx.GraphModule (static model) representing the Pytorch model.
    """
    set_model(pytorch_model)
    symbolic_traced = symbolic_trace(pytorch_model)
    inputs = representative_data_gen()
    input_for_shape_infer = [to_tensor(i) for i in inputs]
    ShapeProp(symbolic_traced).propagate(*input_for_shape_infer)
    return symbolic_traced


def remove_broken_nodes_from_graph(graph):
    """
    Remove all "broken" nodes (nodes without output).

    Args:
        graph: Networkx MultiDiGraph representing the Pytorch model.

    Returns:
        Networkx MultiDiGraph representing the Pytorch model after removing nodes without an output
        (for example: "assert").

    """
    output_nodes = [n.node for n in graph.output_nodes]
    # make a list of all the nodes with no output
    nodes_list = [n for n, d in graph.out_degree if d == 0 and n not in output_nodes]
    nodes_to_remove = []
    for node in nodes_list:
        nodes_to_remove.append(node)
        # iterate over all the relevant nodes
        while len(nodes_to_remove) != 0:
            for node_to_remove in nodes_to_remove:
                parent_nodes = [edge.source_node for edge in graph.incoming_edges(node_to_remove)]
                # check all the parent nodes
                for parent_node in parent_nodes:
                    # if the parent node is connected only to the "broken" node, we'll add the parent node to
                    # the relevant nodes list
                    # if the parent node is connected to other nodes, we'll only remove the edge
                    if graph.out_degree(parent_node) == 1:
                        nodes_to_remove.append(parent_node)
                    graph.remove_edge(parent_node, node_to_remove)
                # remove the current node
                graph.remove_node(node_to_remove)
                nodes_to_remove.remove(node_to_remove)
    return graph


def model_reader(model: torch.nn.Module,
                 representative_data_gen: Callable,
                 to_numpy: Callable,
                 to_tensor: Callable) -> Graph:
    """
    Reads a Pytorch model and converts it to an FX Graph using the fx toolkit. Then, builds a base graph representing
    the fx graph. Finally, we filter "broken nodes" (nodes without outputs, for example: "assert").
    Args:
        model: Pytorch model to build its graph representation.
        representative_data_gen (Callable): Dataset used for calibration.
        to_numpy: Function to convert framework's tensor to a Numpy array.
        to_tensor: Function to convert a Numpy array to a framework's tensor.

    Returns:
        Base graph of the Pytorch model.
    """
    logging.info("Start Model Reading...")
    fx_model = fx_graph_module_generation(model, representative_data_gen, to_tensor)
    graph = build_graph(fx_model, to_numpy)
    graph = remove_broken_nodes_from_graph(graph)
    return graph
