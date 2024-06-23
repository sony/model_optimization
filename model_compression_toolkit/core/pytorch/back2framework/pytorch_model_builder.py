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
from abc import abstractmethod
from functools import partial
from typing import Tuple, Any, Dict, List, Union, Callable

import torch
import numpy as np
from networkx import topological_sort

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.back2framework.base_model_builder import BaseModelBuilder
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER
from mct_quantizers import PytorchQuantizationWrapper


def _build_input_tensors_list(node: BaseNode,
                              graph: Graph,
                              inputs: Tuple[Any],
                              node_to_output_tensors_dict: Dict[BaseNode, List]) -> List[List]:
    """
    Given a node, build a list of input tensors the node gets. The list is built based on the
    node's incoming edges, previous nodes' output tensors.

    Args:
        node: Node to build its input tensors list.
        graph: Graph the node is in.
        inputs: list of input tensors to model.
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.

    Returns:
        A list of the node's input tensors.
    """
    if node.is_match_type(DummyPlaceHolder):
        input_tensors = [inputs[graph.get_inputs().index(node)]]
    else:
        input_tensors = []
        # Go over a sorted list of the node's incoming edges, and for each source node get its output tensors.
        # Append them in a result list.
        for ie in graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX):
            _input_tensors = node_to_output_tensors_dict[ie.source_node]
            input_tensors.append(_input_tensors)
        input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
    return input_tensors


def _merge_inputs(_node: BaseNode, input_tensors: List, op_call_args: List, op_call_kwargs: Dict,
                  tensor_input_allocs: List = None) -> Tuple[List, Dict]:
    """
    Merge input tensors list with positional weights and op_call_args, according to correct order.

    Args:
        _node: The node the inputs are for.
        input_tensors: activation input tensors to node.
        op_call_args: framework node call args.
        op_call_kwargs: framework node call kwargs.
        tensor_input_allocs: List of input allocations to node.

    Returns:
        Combined list of input_tensors and op_call_args.
    """
    if isinstance(_node, FunctionalNode) and _node.tensor_input_allocs:
        _input_list = op_call_args.copy()
        if tensor_input_allocs is None:
            tensor_input_allocs = _node.tensor_input_allocs
        if len(tensor_input_allocs) != len(input_tensors):
            Logger.error(f'Mismatch between input tensors ({len(tensor_input_allocs)}) '
                         f'and indices {len(input_tensors)} in node {_node.name}.')  # pragma: no cover
        for i, t in zip(tensor_input_allocs, input_tensors):
            # insert input tensors in either args or kwargs, according to tensor_input_allocs
            if isinstance(i, str):
                assert i not in op_call_kwargs
                op_call_kwargs.update({i: t})
            else:
                _input_list.insert(i, t)
    else:
        _input_list = input_tensors + op_call_args

    return _input_list, op_call_kwargs


def _run_operation(n: BaseNode,
                   input_tensors: List,
                   op_func: Any,
                   quantize_node_activation_fn,
                   use_activation_quantization: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Applying the layer (op_func) to the input tensors (input_tensors).
    If quantized is set to True, and the layer's corresponding node (n) has quantization
    attributes, an additional fake-quantization node is built and appended to the layer.

    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of Pytorch tensors that are the layer's inputs.
        op_func: Module/functional to apply to the input tensors.
        quantize_node_activation_fn: quantization function
        use_activation_quantization: Flag to indicate if we have an activation function.
    Returns:
        A tuple of Pytorch tensors. The Module/functional output tensors after applying the
        Module/functional to the input tensors.
    """

    op_call_args = n.op_call_args if isinstance(n, FunctionalNode) else []
    functional_kwargs = n.op_call_kwargs if isinstance(n, FunctionalNode) else {}

    # Insert positional weights only when not a quantized functional node, because quantized functional nodes
    # insert the quantized weights in the wrapper.
    if isinstance(n, FunctionalNode) and isinstance(op_func, PytorchQuantizationWrapper):
        _tensor_input_allocs = [i for i in n.tensor_input_allocs if i not in n.weights]
    else:
        input_tensors = n.insert_positional_weights_to_input_list(input_tensors)
        # convert inputs from positional weights (numpy arrays) to tensors. Must handle each element in the
        # list separately, because in FX the tensors are FX objects and fail to_torch_tensor
        input_tensors = [to_torch_tensor(t, numpy_type=t.dtype) if isinstance(t, np.ndarray) else t
                         for t in input_tensors]
        _tensor_input_allocs = None

    if isinstance(n, FunctionalNode) and n.inputs_as_list:
        out_tensors_of_n_float = op_func(input_tensors, *op_call_args, **functional_kwargs)
    else:
        merged_inputs, functional_kwargs = _merge_inputs(n, input_tensors, op_call_args, functional_kwargs.copy(),
                                                         tensor_input_allocs=_tensor_input_allocs)
        out_tensors_of_n_float = op_func(*merged_inputs, **functional_kwargs)

    # Add a fake quant node if the node has an activation threshold.
    out_tensors_of_n = out_tensors_of_n_float
    if use_activation_quantization:
        if isinstance(out_tensors_of_n_float, list):
            out_tensors_of_n_float = torch.cat(out_tensors_of_n_float, dim=0)
        out_tensors_of_n = quantize_node_activation_fn(out_tensors_of_n_float)

    if not isinstance(out_tensors_of_n, list):
        out_tensors_of_n, out_tensors_of_n_float = [out_tensors_of_n], [out_tensors_of_n_float]
    return out_tensors_of_n, out_tensors_of_n_float


def _find_by_node_name(node_to_output_tensors_dict: dict, node_name: str):
    """
    Args:
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.
        node_name: Node name
    Returns:
        Out tensors if found node with name equal to node_name, None if no exist
    """
    for node in node_to_output_tensors_dict.keys():
        if node.name == node_name:
            return node_to_output_tensors_dict.get(node)
    return None


def _generate_outputs(
        out_nodes: List[BaseNode],
        node_to_output_tensors_dict: dict):
    """
    Args:
        out_nodes: List of output nodes.
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.

    Returns:
        List of output tensor/s for the model
    """
    output = []
    for n in out_nodes:
        out_tensors_of_n = _find_by_node_name(node_to_output_tensors_dict, n.name)
        if len(out_tensors_of_n) > 1:
            output.append(out_tensors_of_n)
        else:
            output += out_tensors_of_n
    return output


class PytorchModel(torch.nn.Module):
    """
    Class for reconstructing a Pytorch model from a graph
    """

    def __init__(self,
                 graph: Graph,
                 append2output: List[Any] = None,
                 return_float_outputs: bool = False,
                 wrapper: Callable = None,
                 get_activation_quantizer_holder_fn: Callable = None):
        """
        Construct a Pytorch model.

        Args:
            graph: Graph to build its corresponding Pytorch model.
            append2output: List of nodes or OutTensor objects.
            return_float_outputs: Whether the model returns float tensors or not.
            wrapper: A function wrapper Pytorch Layers.
            get_activation_quantizer_holder_fn: Function to retrieve a quantization holder for a node.

        """
        super(PytorchModel, self).__init__()
        self.graph = graph
        self.node_sort = list(topological_sort(graph))
        self.node_to_activation_quantization_holder = {}
        self.append2output = append2output
        self.return_float_outputs = return_float_outputs
        self.wrapper = wrapper
        self.get_activation_quantizer_holder = get_activation_quantizer_holder_fn
        self._add_modules()

    # todo: Move to parent class BaseModelBuilder
    @property
    def use_activation_holder_during_model_building(self) -> bool:
        """
        Returns: Whether or not the model builder uses a PytorchActivationQuantizationHolder during
        model building (by adding it as a module when converting the graph to a Pytorch model).
        If so - the model builder expects the activation quantizers not to be wrapped
        in a PytorchQuantizeWrapper.
        """
        return self.get_activation_quantizer_holder is not None

    @abstractmethod
    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """
        raise NotImplemented(f'{self.__class__.__name__} '
                             f'have to implement a method for quantization activation nodes.')  # pragma: no cover

    def wrap(self, node):
        """
        Wraps a node operation with a wrapper, if one is available.

        Args:
            node: node to wrap its operation.

        Returns: the node's operation. If a wrapper is available, the operation is wrapped.
        """
        if isinstance(node, FunctionalNode):
            if self.wrapper is None:
                node_op = node.type
            else:
                node_op = self.wrapper(node, node.type)
        else:
            if self.wrapper is None:
                node_op = node_builder(node)
            else:
                node_op = self.wrapper(node, node_builder(node))
        return node_op

    def _add_modules(self):
        """
        Build and add the modules and functional nodes from node_sort list as attributes to PytorchModel
        """
        for node in self.node_sort:
            node_op = self.wrap(node)
            if isinstance(node, FunctionalNode):
                # for functional layers
                setattr(self, node.name, node_op)
            else:
                self.add_module(node.name, node_op)

            # Add activation quantization modules if an activation holder is configured for this node
            if node.is_activation_quantization_enabled() and self.get_activation_quantizer_holder is not None:
                activation_quantizer_holder = self.get_activation_quantizer_holder(node)
                if activation_quantizer_holder is not None:
                    self.add_module(node.name + '_' + ACTIVATION_HOLDER_QUANTIZER, activation_quantizer_holder)
                    self.node_to_activation_quantization_holder.update(
                        {node.name: node.name + '_' + ACTIVATION_HOLDER_QUANTIZER})

    def forward(self,
                *args: Any) -> Any:
        """
        Args:
            args: argument input tensors to model.
        Returns:
            torch Tensor/s which is/are the output of the model logic.
        """
        node_to_output_tensors_dict = dict()
        node_to_output_tensors_dict_float = dict()
        configurable_nodes = self.graph.get_configurable_sorted_nodes_names(DEFAULT_PYTORCH_INFO)
        for node in self.node_sort:
            op_func = self._get_op_func(node, configurable_nodes)
            input_tensors = _build_input_tensors_list(node,
                                                      self.graph,
                                                      args,
                                                      node_to_output_tensors_dict)
            use_activation_quantization, activation_quantization_fn = self._get_activation_quantization_fn(node)

            # Run node operation and fetch outputs
            out_tensors_of_n, out_tensors_of_n_float = _run_operation(node,
                                                                      input_tensors,
                                                                      op_func=op_func,
                                                                      quantize_node_activation_fn=activation_quantization_fn,
                                                                      use_activation_quantization=use_activation_quantization)

            node_to_output_tensors_dict.update({node: out_tensors_of_n})
            node_to_output_tensors_dict_float.update({node: out_tensors_of_n_float})

        if self.append2output:
            outputs = _generate_outputs(self.append2output,
                                        node_to_output_tensors_dict_float if self.return_float_outputs else node_to_output_tensors_dict)
        else:
            outputs = _generate_outputs([ot.node for ot in self.graph.get_outputs()],
                                        node_to_output_tensors_dict_float if self.return_float_outputs else node_to_output_tensors_dict)
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs

    def _get_op_func(self,
                     node: BaseNode,
                     configurable_nodes_names: List[str]) -> Any:
        """
        Gets the operation function that runs the actual inference of the nodes compatible layer.

        Args:
            node: The corresponding node of the layer it runs.
            configurable_nodes_names: A list of names of configurable nodes in the quantized model.

        Returns: Module/functional to apply to the input tensors.

        """
        return getattr(self, node.name)

    def _get_activation_quantization_fn(self, node) -> Tuple[bool, Callable]:
        """
        Get activation quantization parameters for this node.

        Args:
            node: Node from which to extract the activation quantization parameters.

        Returns:
            Flag to indicate if we quantize activations using a quantization holder and a quantization
            function to use for the node's activations.
        """
        activation_quantization_holder = self.node_to_activation_quantization_holder.get(node.name)
        use_activation_quantization = node.is_activation_quantization_enabled()
        if use_activation_quantization:
            if activation_quantization_holder is None:
                activation_quantization_fn = partial(self._quantize_node_activations, node)
                use_activation_quantization = self.wrapper is None
            else:
                activation_quantization_fn = getattr(self, activation_quantization_holder)
        else:
            activation_quantization_fn = None
        return use_activation_quantization, activation_quantization_fn


class PyTorchModelBuilder(BaseModelBuilder):
    """
    Builder of PyTorch models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                 return_float_outputs: bool = False,
                 wrapper: Callable = None,
                 get_activation_quantizer_holder_fn: Callable = None):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
            wrapper: A function wrapper Pytorch Layers.
            get_activation_quantizer_holder_fn: Function to retrieve a quantization holder for a node.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

        self.wrapper = wrapper
        self.get_activation_quantizer_holder_fn = get_activation_quantizer_holder_fn

    def build_model(self) -> Tuple[PytorchModel, UserInformation]:
        """
        Build a PyTorch model and return it.
        Returns: Pytorch model and user information.

        """
        return PytorchModel(self.graph,
                            self.append2output,
                            return_float_outputs=self.return_float_outputs,
                            wrapper=self.wrapper,
                            get_activation_quantizer_holder_fn=self.get_activation_quantizer_holder_fn), self.graph.user_info
