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
import inspect
from typing import Dict, List, Tuple, Callable
import torch
from torch.fx import GraphModule

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import OUTPUT, PLACEHOLDER, TENSOR_META, CALL_FUNCTION, TYPE, \
    CALL_METHOD, BIAS, FUNCTIONAL_OP, OP_CALL_KWARGS, OP_CALL_ARGS, INPUTS_AS_LIST, GET_ATTR, CONSTANT, BUFFER
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder, ConstantHolder, BufferHolder
from model_compression_toolkit.logger import Logger


def extract_holder_weights(constant_name, node_target, model, weights, to_numpy):
    """
    Extract layer weights and named buffers for BufferHolder and ConstantHolder.
    Args:
        constant_name: name to write the parameters under, CONSTANT for ConstantHolder and
         BUFFER for BufferHolder.
        node_target: relevant parameter name from Pytorch FX model.
        model: Pytorch FX model.
        weights: dictionary containing the weights of the node.
        to_numpy: Function to convert framework's tensor to a Numpy array.

    Returns:
        Updated weights dictionary.
    """
    named_parameters_weights = {constant_name: to_numpy(parameter) for name, parameter in
                                model.named_parameters() if node_target == name}
    named_buffer_weights = {constant_name: to_numpy(parameter) for name, parameter in
                            model.named_buffers() if node_target == name}
    if len(named_parameters_weights) + len(named_buffer_weights) > 1:
        raise Exception(
            f'Constant parameter can only have one tensor. Here we have '
            f'{len(named_parameters_weights)+ len(named_buffer_weights)}')

    weights.update(named_parameters_weights)
    weights.update(named_buffer_weights)
    return weights


def nodes_builder(model: GraphModule,
                  module_dict: Dict,
                  to_numpy: Callable) -> Tuple[List, List, List, Dict]:
    """
    Build a node from a fx node. A node contains all information to reconstruct the model module or call function
    it's representing in the model: operation, module configuration, weights, input/output shape.
    Args:
        model: Pytorch FX model.
        module_dict: A dictionary of the Pyotrch model's named modules.
        to_numpy: A function to convert a Tensor to numpy array

    Returns:
        A list of Graph nodes that were built from the fx GraphModule nodes.
    """
    # init function variables:
    inputs = []
    outputs = []
    nodes = []
    output_nodes = []
    fx_node_2_graph_node = {}

    for node in model.graph.nodes:
        # extract node type and framework attributes
        framework_attr = dict(node.kwargs)
        node_has_activation = True
        if node.target in module_dict.keys():
            node_module = module_dict[node.target]
            node_type = type(node_module)
            framework_attr = node_module.__dict__
            fullargspec = inspect.getfullargspec(node_type.__init__).args
            framework_attr = {k: v for k, v in framework_attr.items() if k in fullargspec}
            if hasattr(node_module, BIAS) and BIAS in fullargspec:
                framework_attr[BIAS] = False if node_module.bias is None else True
        elif node.op == CALL_FUNCTION:
            node_type = node.target
            if node_type == getattr:
                node_has_activation = False
                Logger.warning(
                    'Pytorch model has a parameter or constant Tensor value. This can cause unexpected behaviour when '
                    'converting the model.')
        elif node.op == PLACEHOLDER:
            node_type = DummyPlaceHolder
        elif node.op == OUTPUT:
            output_nodes += node.all_input_nodes
            continue
        elif node.op == CALL_METHOD:
            if hasattr(torch, node.target):
                node_type = getattr(torch, node.target)
            elif hasattr(torch.Tensor, node.target):
                node_type = getattr(torch.Tensor, node.target)
            else:
                raise Exception(f'Call method of type \'{node.target}\' is currently not supported.')
        elif node.op == GET_ATTR:
            if node.meta[TYPE] == torch.Tensor:
                node_type = BufferHolder
            else:
                node_type = ConstantHolder
            node_has_activation = False
            Logger.warning(
                'Pytorch model has a parameter or constant Tensor value. This can cause unexpected behaviour when '
                'converting the model.')
        else:
            raise Exception(f'Unknown node type: {node.name}')

        # extract layer weights and named buffers
        weights = {}
        if node.target in module_dict.keys():
            named_parameters_weights = {name: to_numpy(parameter) for name, parameter in
                                        module_dict[node.target].named_parameters()}
            named_buffer_weights = {name: to_numpy(parameter) for name, parameter in
                                    module_dict[node.target].named_buffers() if len(parameter.shape) > 0}
            weights.update(named_parameters_weights)
            weights.update(named_buffer_weights)

        if node.op == GET_ATTR:
            if node_type == ConstantHolder:
                weights = extract_holder_weights(CONSTANT, node.target, model, weights, to_numpy)
                framework_attr.update(const_size=weights.get(CONSTANT).shape)
            elif node_type == BufferHolder:
                weights = extract_holder_weights(BUFFER, node.target, model, weights, to_numpy)
                framework_attr.update(name=node.name)

        # extract input shapes
        input_shape = []
        if node.op != PLACEHOLDER:
            for input_node in node.all_input_nodes:
                tensor_meta = input_node.meta
                if tensor_meta[TYPE] == torch.Tensor:
                    input_shape += [list(tensor_meta[TENSOR_META].shape)]
                elif tensor_meta[TYPE] == tuple:
                    input_shape += [list(n.shape) for n in tensor_meta[TENSOR_META]]
                elif tensor_meta[TYPE] == int:
                    input_shape += [[1]]

        # extract output shapes
        if node.meta[TYPE] == torch.Tensor:
            output_shape = [list(node.meta[TENSOR_META].shape)]
        elif node.meta[TYPE] in (list, tuple):
            output_shape = [list(m.shape) for m in node.meta[TENSOR_META]]
        elif node.meta[TYPE] == int:
            output_shape = [[1]]
        else:
            output_shape = []

        # filter Nodes from framework attributes, we replace these attributes with nx graph nodes
        framework_attr_filtered = {}
        for k, v in framework_attr.items():
            if not isinstance(v, torch.fx.node.Node):
                framework_attr_filtered[k] = v
        framework_attr = framework_attr_filtered

        # filter Nodes from node kwargs, we replace these attributes with nx graph nodes
        node_kwargs = {}
        for k, v in node.kwargs.items():
            if not isinstance(v, torch.fx.node.Node):
                node_kwargs[k] = v

        # initiate graph nodes
        if node.op in [CALL_METHOD, CALL_FUNCTION]:
            graph_node_type = FunctionalNode
            inputs_as_list1 = len(node.args) > 0 and isinstance(node.args[0], (list, tuple)) and all(
                [isinstance(n, torch.fx.node.Node) for n in node.args[0]])
            inputs_as_list = inputs_as_list1 or \
                             (len(node.args) > 0 and node.args[0].op == PLACEHOLDER and node.args[0].meta[TYPE] in (list, tuple))
            if inputs_as_list:
                num_inputs = 1
            else:
                input_counter = 0
                for in_node in node.all_input_nodes:
                    for arg in node.args:
                        if arg == in_node:
                            input_counter += 1
                num_inputs = max(len(node.all_input_nodes), input_counter)
            op_call_args = list(node.args[num_inputs:])

            # remove torch.fx.node.Node from inputs to graph_node_type
            for arg in op_call_args:
                if isinstance(arg, torch.fx.node.Node):
                    op_call_args.remove(arg)

            kwargs = {FUNCTIONAL_OP: node_type,
                      OP_CALL_ARGS: op_call_args,
                      OP_CALL_KWARGS: node_kwargs,
                      INPUTS_AS_LIST: inputs_as_list}
        else:
            graph_node_type = BaseNode
            kwargs = {}
        graph_node = graph_node_type(name=node.name,
                                     framework_attr=framework_attr,
                                     input_shape=input_shape,
                                     output_shape=output_shape,
                                     weights=weights,
                                     layer_class=node_type,
                                     has_activation=node_has_activation,
                                     **kwargs)

        # generate graph inputs list
        if node.op == PLACEHOLDER:
            for ii in range(len(output_shape)):
                inputs.append(graph_node)

        fx_node_2_graph_node[node] = graph_node
        nodes.append(graph_node)

    # generate graph outputs list
    for node in output_nodes:
        outputs.append(OutTensor(fx_node_2_graph_node[node], output_nodes.index(node)))

    return nodes, inputs, outputs, fx_node_2_graph_node


def edges_builder(model: GraphModule,
                   fx_node_2_graph_node: Dict) -> List:
    """

    Args:
        model: Pytorch FX model.
        fx_node_2_graph_node: dictionary from fx node to graph node.

    Returns:
        List of graph edges
    """
    src_index = 0 # in fx src_index is always zero because fx uses the getitem operator to fetch node outputs
    edges = []
    connectivity_dict = {}
    for node in model.graph.nodes:
        if node.op != OUTPUT:
            for input_node in node.all_input_nodes:

                # n_edges_for_input_node is for the case that the input node appears more than
                # once as the input of the node, for example add(x, x)
                n_edges_for_input_node = sum([1 for a in node.args if input_node == a])
                n_edges_for_input_node = max(n_edges_for_input_node, 1)

                dst_index = node.all_input_nodes.index(input_node)
                for i in range(n_edges_for_input_node):
                    if connectivity_dict.get(input_node):
                        connectivity_dict[input_node].append((node, dst_index))
                    else:
                        connectivity_dict[input_node] = [(node, dst_index)]
                    dst_index += 1
    for node in model.graph.nodes:
        out_nodes = connectivity_dict.get(node)
        if out_nodes:
            for (out_node, dst_index) in out_nodes:
                edges.append(
                    Edge(fx_node_2_graph_node[node], fx_node_2_graph_node[out_node], src_index, dst_index))

    return edges