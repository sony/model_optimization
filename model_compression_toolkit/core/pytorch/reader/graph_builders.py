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
from operator import getitem
from typing import Dict, List, Tuple, Callable, Union, Any, Type

import numpy as np
import torch
from torch.fx import GraphModule, Node

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import OUTPUT, PLACEHOLDER, TENSOR_META, CALL_FUNCTION, TYPE, \
    CALL_METHOD, BIAS, FUNCTIONAL_OP, OP_CALL_KWARGS, OP_CALL_ARGS, INPUTS_AS_LIST, TENSOR_INPUT_ALLOCS, GET_ATTR
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.logger import Logger


def _extract_parameters_and_buffers(module: Union[torch.nn.Module, GraphModule]) -> Dict[str, np.ndarray]:
    """
    Extract parameters & buffers from input module to a dictionary.
    Args:
        module: FX ot PyTorch module to extract parameters and buffers from.

    Returns:
        Dictionary containing module parameters and buffers by name.
    """

    named_parameters = {name: parameter for name, parameter in module.named_parameters()}
    named_buffers = {name: buffer for name, buffer in module.named_buffers()}

    return {**named_parameters, **named_buffers}


def is_instance_first_arg(n: Node, expected_type: Union[Type, Tuple[Type]]) -> bool:
    """
    Check whether first argument of the node is the expected type
    Args:
        n: fx node.
        expected_type: Expected 1st argument type.

    Returns:
        True is the first argument of node n is of the expected type, else return False.

    """
    return len(n.args) > 0 and isinstance(n.args[0], expected_type)


def _build_input_alloc_and_call_args(n: Node, input_tensors_in_node_kwargs: Dict,
                                     inputs_as_list: bool) -> Tuple[List, List]:
    """
    Build the tensor inputs list and op_call_args of the functional node.

    Args:
        n: fx node.
        input_tensors_in_node_kwargs: A dictionary of node kwarg name and input fx node.
        inputs_as_list: Is node's inputs are a list.

    Returns:
        A list of updated op_call args.
        A list of tensor allocations in node's inputs.

    """

    tensor_input_alloc = []
    op_call_args = list(n.args)
    if inputs_as_list:
        # input tensors are a list in the first argument -> remove from op_call_args and go over
        # the tensors in that list.
        _args = op_call_args.pop(0)
    else:
        _args = n.args
    for in_node in n.all_input_nodes:
        # The extra for loop is used to tackle the case of the same input tensor for this node (e.g. torch.add(x, x)).
        for i, arg in enumerate(_args):
            if arg == in_node:
                tensor_input_alloc.append(i)
        for k, arg in input_tensors_in_node_kwargs.items():
            if arg == in_node:
                tensor_input_alloc.append(k)

    return op_call_args, tensor_input_alloc


def _extract_torch_layer_data(node_module: torch.nn.Module) -> Tuple[Any, Dict[str, np.ndarray], Dict]:
    """
    Extract required data from a non-functional node to rebuild the PyTorch layer.

    Args:
        node_module: Torch layer, such as nn.Conv2d, nn.Linear, etc.

    Returns:
        Node layer class.
        A mapping between the layer's named parameters and buffers to their tensor values.
        A framework_attr dictionary required to instantiate the node with the layer class.
    """
    node_type = type(node_module)
    if not isinstance(node_module, torch.nn.Module):
        Logger.error(f"Expected an instance of torch.nn.Module for node {node_module.name}, but got {node_type}")  # pragma: no cover
    # Extract the instance framework_attr (i.e. the arguments the class instance was initialized with). "fullargspec"
    # is a list of the layer's attribute names, that will be used as keys of the framework_attr dictionary. We the
    # values from the layer instance.
    fullargspec = inspect.getfullargspec(node_type.__init__).args
    framework_attr = {k: v for k, v in node_module.__dict__.items() if k in fullargspec}
    # The "bias" argument doesn't appear in the node_module.__dict__, so we add it manually.
    if hasattr(node_module, BIAS) and BIAS in fullargspec:
        framework_attr[BIAS] = False if node_module.bias is None else True

    # Extract layer weights and named buffers.
    weights = {n: w for n, w in _extract_parameters_and_buffers(node_module).items() if len(w.shape) > 0}
    return node_type, weights, framework_attr


def _extract_input_and_output_shapes(_node: Node) -> Tuple[List, List]:
    """
    Extract input and output shapes of a node.
    Args:
        _node: fx node.

    Returns:
        Input and output shapes as lists.
    """
    input_shape = []
    if _node.op != PLACEHOLDER:
        for i, input_node in enumerate(_node.all_input_nodes):
            tensor_meta = input_node.meta
            if tensor_meta[TYPE] in [torch.Tensor, torch.nn.parameter.Parameter]:
                input_shape += [list(tensor_meta[TENSOR_META].shape)]
            elif tensor_meta[TYPE] == tuple:
                input_shape += [list(n.shape) for n in tensor_meta[TENSOR_META]]
            elif tensor_meta[TYPE] == int:
                input_shape += [[1]]

    if _node.meta[TYPE] == torch.Tensor:
        output_shape = [list(_node.meta[TENSOR_META].shape)]
    elif _node.meta[TYPE] == torch.Size:
        output_shape = [[len(input_shape[0])]] if len(input_shape) > 0 else [[]]
    elif _node.meta[TYPE] in (list, tuple):
        output_shape = [list(m.shape) for m in _node.meta.get(TENSOR_META, [])]
    elif _node.meta[TYPE] in [int, bool]:
        output_shape = [[1]]
    else:
        output_shape = [[]]

    return input_shape, output_shape


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
    # Init function variables:
    inputs, outputs = [], []
    nodes, output_nodes = [], []
    fx_node_2_graph_node = {}
    consts_dict = {}
    used_consts = set()

    # Dictionary to track seen targets and their corresponding nodes to mark reused nodes
    seen_targets = {}

    # Init parameters & buffers dictionary of the entire model. We later extract the constants values from this dictionary.
    model_parameters_and_buffers = _extract_parameters_and_buffers(model)

    for node in model.graph.nodes:

        # ##############################################
        #  Extract node type and framework attributes  #
        # ##############################################
        weights = {}
        framework_attr = {}
        node_has_activation = True

        if node.target in module_dict.keys():
            # PyTorch module node, such as nn.Conv2d or nn.Linear.
            node_type, weights, framework_attr = _extract_torch_layer_data(module_dict[node.target])

        elif node.op == CALL_FUNCTION:
            # Node is a function that handle a parameter\buffer in the model.
            node_type = node.target
            if node_type in [getattr, getitem]:
                node_has_activation = False

        elif node.op == PLACEHOLDER:
            # Input node to the model.
            node_type = DummyPlaceHolder

        elif node.op == OUTPUT:
            # Output node of the model. Only saved in output_nodes for later handling.
            output_nodes += node.all_input_nodes
            continue

        elif node.op == CALL_METHOD:
            # Node is a PyTorch function such as torch.add, torch.reshape etc.
            if hasattr(torch, node.target):
                node_type = getattr(torch, node.target)
            elif hasattr(torch.Tensor, node.target):
                node_type = getattr(torch.Tensor, node.target)
                if node_type==torch.Tensor.to:
                    Logger.critical(f"The call method \"to\" is not supported. Please consider moving \"torch.Tensor.to\" operations to init code.")  # pragma: no cover
            else:
                Logger.critical(f"The call method '{node.target}' in {node} is not supported.")  # pragma: no cover

        elif node.op == GET_ATTR:
            # Node holding a constant -> add to consts_dict so can add them later to weights of next node.
            if node.target in consts_dict:
                Logger.critical('A constant weight appears to have been recorded multiple times.')  # pragma: no cover
            consts_dict[node] = model_parameters_and_buffers[node.target]
            continue
        else:
            Logger.critical(f'Encountered an unsupported node type in node: {node.name}.')  # pragma: no cover

        # Add constants to weights dictionary.
        if node.op != PLACEHOLDER:
            if len(node.args) and isinstance(node.args[0], (list, tuple)):
                # handle weights in nodes with list input. Especially when there's a duplicate of a tensor
                # in the input list (e.g. torch.concat([const1, x, const2, x, const3], 1)).
                for input_node in node.all_input_nodes:
                    for i, input_arg in enumerate(node.args[0]):
                        if input_node is input_arg and input_node in consts_dict:
                            used_consts.add(input_node)
                            weights.update({i: consts_dict[input_node]})
            else:
                for i, input_node in enumerate(node.all_input_nodes):
                    if input_node in consts_dict:
                        used_consts.add(input_node)
                        weights.update({i: consts_dict[input_node]})

        # Extract input and output shapes of the node.
        input_shape, output_shape = _extract_input_and_output_shapes(node)

        # Check if this node's target has been seen before
        reuse = False
        reuse_group = None
        node_group_key = create_reuse_group(node.target, weights)
        # We mark nodes as reused only if there are multiple nodes in the graph with same
        # 'target' and it has some weights.
        if node_group_key in seen_targets and len(weights) > 0:
            reuse = True
            reuse_group = node_group_key
            # Update the 'base/main' node with the reuse group as all other nodes in its group.
            fx_node_2_graph_node[seen_targets[node_group_key]].reuse_group = reuse_group
        else:
            seen_targets[node_group_key] = node

        # Convert weights to numpy arrays after reuse marking
        # We delay this conversion to preserve the original tensor instances during the reuse identification process.
        # This is crucial for correctly identifying identical weight instances in reused functional layers.
        # By keeping the original PyTorch tensors until this point, we ensure that:
        # 1. Reused layers with the same weight instances are correctly marked as reused.
        # 2. The instance-based weight signature generation works as intended, using the memory
        # addresses of the original tensors.
        # Only after all reuse marking is complete do we convert to numpy arrays.
        for weight_name, weight_value in weights.items():
            weights[weight_name] = to_numpy(weight_value)

        # Initiate graph nodes.
        if node.op in [CALL_METHOD, CALL_FUNCTION]:
            graph_node_type = FunctionalNode

            # Filter FX nodes from node_kwargs. These FX nodes are tensor inputs to the node that are part of the
            # model's graph. We remove them because the node_kwargs should not include input tensors of the node.
            # These input tensors will be inserted in the kwargs according to the tensor_input_alloc which is used
            # to convert the input_tensors list in the builder to the node's args & kwargs.
            node_kwargs, input_tensors_in_node_kwargs = {}, {}
            for k, v in node.kwargs.items():
                if isinstance(v, Node):
                    input_tensors_in_node_kwargs[k] = v
                else:
                    node_kwargs[k] = v

            # Check if node's first input argument is a list of input fx nodes, such as torch.cat:
            inputs_as_list = is_instance_first_arg(node, (list, tuple)) and all(
                [isinstance(n, Node) for n in node.args[0]])

            # Build tensor_input_alloc required for the model builder. All input nodes are received as a list in the builder,
            # so tensor_input_alloc is used to allocate each input tensor in the correct place in the node's args & kwargs.
            op_call_args, tensor_input_alloc = _build_input_alloc_and_call_args(node, input_tensors_in_node_kwargs,
                                                                                inputs_as_list)

            # Remove torch.fx.node.Node from inputs to the functional node. FX nodes are input tensors in the builder,
            # so they are remove from the op_call_args (same as op_call_kwargs) and are inserted back according to the
            # tensor_input_alloc list.
            op_call_args = [arg for arg in op_call_args if not isinstance(arg, Node)]
            # Convert torch.fx.immutable_collections.immutable_list to tuple.
            op_call_args = [tuple(arg) if isinstance(arg, torch.fx.immutable_collections.immutable_list) else arg
                            for arg in op_call_args]

            kwargs = {FUNCTIONAL_OP: node_type,
                      OP_CALL_ARGS: op_call_args,
                      OP_CALL_KWARGS: node_kwargs,
                      INPUTS_AS_LIST: inputs_as_list,
                      TENSOR_INPUT_ALLOCS: tensor_input_alloc}
        else:
            if not all([not isinstance(v, Node) for v in framework_attr.values()]):
                Logger.critical(f'Found FX nodes in framework attributes of {node.name}. This node type should not contain any.')  # pragma: no cover

            graph_node_type = BaseNode
            kwargs = {}

        graph_node = graph_node_type(name=node.name,
                                     framework_attr=framework_attr,
                                     input_shape=input_shape,
                                     output_shape=output_shape,
                                     weights=weights,
                                     layer_class=node_type,
                                     has_activation=node_has_activation,
                                     reuse=reuse,
                                     reuse_group=reuse_group,
                                     **kwargs)

        # Generate graph inputs list.
        if node.op == PLACEHOLDER:
            for ii in range(len(output_shape)):
                inputs.append(graph_node)

        fx_node_2_graph_node[node] = graph_node
        nodes.append(graph_node)

    # Check whether all extracted constants were used in the graph.
    not_connected_consts = [c for c in consts_dict if c not in used_consts]
    if not_connected_consts:
        Logger.critical(f'Error reading graph: These constants are not connected in the graph: {not_connected_consts}.')  # pragma: no cover

    # Generate graph outputs list.
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
    src_index = 0  # in fx src_index is always zero because fx uses the getitem operator to fetch node outputs
    edges = []
    connectivity_dict = {}
    for node in model.graph.nodes:
        if node.op != OUTPUT:
            for input_node in node.all_input_nodes:
                if input_node in fx_node_2_graph_node:
                    # n_edges_for_input_node is for the case that the input node appears more than
                    # once as the input of the node, for example add(x, x)
                    if node in fx_node_2_graph_node and isinstance(fx_node_2_graph_node[node], FunctionalNode) and \
                            fx_node_2_graph_node[node].inputs_as_list:
                        _args = node.args[0]
                    else:
                        _args = node.args
                    n_edges_for_input_node = sum([1 for a in _args if input_node == a])
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


def create_reuse_group(target: Any, weights: Dict[str, Any]) -> str:
    """
    Combine target and weights to create a unique reuse group identifier.
    We consider the weights as part of the group identifier because they are not part of
    the module in functional layers, but if a functional layer is using the same weights multiple
    times it is considered to be reused.

    This function creates a unique string identifier for a reuse group by combining
    the target (typically a layer or operation name) with the weights IDs.

    Args:
        target (Any): The target of the node, typically a string or callable representing
                      a layer or operation.
        weights (Dict[str, Any]): A dictionary of weight names to weight values.
                                  The values can be any type (typically tensors or arrays).

    Returns:
        str: A unique string identifier for the reuse group.
    """
    if not weights:
        return str(target)
    weight_ids = tuple(sorted(id(weight) for weight in weights.values()))
    return f"{target}_{weight_ids}"
