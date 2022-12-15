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


import logging
from typing import Callable, Dict

import torch
from torch.fx import GraphModule, Tracer
from torch.fx.passes.shape_prop import ShapeProp

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.pytorch.reader.graph_builders import edges_builder, nodes_builder
from model_compression_toolkit.core.pytorch.utils import set_model


class LeafModulesTracer(Tracer):
    """
    LeafModulesTracer overrides is_leaf_module method to make symbolic trace
    model contains layers, using inputs variables to control flow.
    """

    def __init__(self, *args, leaf_modules=None, **kwargs):
        super().__init__(*args, **kwargs)
        if leaf_modules is not None:
            self.leaf_modules = tuple(leaf_modules)
        else:
            self.leaf_modules = ()

    def is_leaf_module(self,
                       m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        We override this method to specify the user's custom layers as "leaf" module.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.leaf_modules and isinstance(m, self.leaf_modules):
            return True
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def symbolic_trace_leaf_modules_aware(pytorch_model: torch.nn.Module,
                                      model_leaf_layers: list = None) -> torch.fx.GraphModule:
    """
    Symbolic tracing with layers using inputs variables to control flow.

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    Args:
        pytorch_model: A dynamic PyTorch model.
        model_leaf_layers (list): List of the module's custom layers,using inputs variables
        to control flow, these layers shouldn't be divided into their submodules.
        Please note that their quantization will not be optimized.

    Returns:
        fx.GraphModule: a Module created from the recorded operations from ``root``.
    """

    tracer = LeafModulesTracer(leaf_modules=model_leaf_layers)
    try:
        graph = tracer.trace(pytorch_model)
    except Exception:
        raise Exception("Float model contains layers that used inputs variables to control flow, please trace these "
                        "layers and pass them as a list in the model_leaf_layers attribute in the "
                        "pytorch_post_training_quantization API.\n These layers shouldn't be divided into their "
                        "submodules.\n Please note that their quantization will not be optimized.")
    name = pytorch_model.__class__.__name__ if isinstance(pytorch_model, torch.nn.Module) else pytorch_model.__name__
    return GraphModule(tracer.root, graph, name)


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
                               to_tensor: Callable,
                               model_leaf_layers: list = None) -> torch.fx.GraphModule:
    """
    Generates a fx.GraphModule from a torch.nn.Module.

    Args:
        pytorch_model: A dynamic PyTorch model.
        representative_data_gen (Callable): Representative dataset used for shape inference.
        to_tensor: Function to convert a Numpy array to a framework's tensor.
        model_leaf_layers (list): List of the module's custom layers,using inputs variables
        to control flow, these layers shouldn't be divided into their submodules.
        Please note that their quantization will not be optimized.

    Returns:
        A fx.GraphModule (static model) representing the PyTorch model.
    """
    set_model(pytorch_model)
    symbolic_traced = symbolic_trace_leaf_modules_aware(pytorch_model=pytorch_model,
                                                        model_leaf_layers=model_leaf_layers)
    inputs = next(representative_data_gen())
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
                 to_tensor: Callable,
                 model_leaf_layers: list = None) -> Graph:
    """
    Reads a Pytorch model and converts it to an FX Graph using the fx toolkit. Then, builds a base graph representing
    the fx graph. Finally, we filter "broken nodes" (nodes without outputs, for example: "assert").
    Args:
        model: Pytorch model to build its graph representation.
        representative_data_gen (Callable): Dataset used for calibration.
        to_numpy: Function to convert framework's tensor to a Numpy array.
        to_tensor: Function to convert a Numpy array to a framework's tensor.
        model_leaf_layers (list): List of the module's custom layers, these layers shouldn't be divided into
        their submodules.
        Please note that their quantization will not be optimized.

    Returns:
        Base graph of the Pytorch model.
    """
    logging.info("Start Model Reading...")
    fx_model = fx_graph_module_generation(model, representative_data_gen, to_tensor,
                                          model_leaf_layers=model_leaf_layers)
    graph = build_graph(fx_model, to_numpy)
    graph = remove_broken_nodes_from_graph(graph)
    return graph
