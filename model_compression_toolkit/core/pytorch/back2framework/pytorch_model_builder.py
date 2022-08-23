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
from typing import Tuple, Any, Dict, List, Union

import torch
from networkx import topological_sort

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.back2framework.base_model_builder import BaseModelBuilder
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder, BufferHolder
from model_compression_toolkit.core.pytorch.utils import get_working_device
from model_compression_toolkit.core.pytorch.constants import BUFFER


def _build_input_tensors_list(node: BaseNode,
                              graph: Graph,
                              inputs: Tuple[Any],
                              node_to_output_tensors_dict: Dict[BaseNode, List]) -> List[List]:
    """
    Given a node, build a list of input tensors the node gets. The list is built
    based on the node's incoming edges and previous nodes' output tensors.

    Args:
        node: Node to build its input tensors list.
        graph: Graph the node is in.
        inputs: list of input tensors to model
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.

    Returns:
        A list of the node's input tensors.
    """
    if node.type == DummyPlaceHolder:
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


def _run_operation(n: BaseNode,
                   input_tensors: List,
                   op_func: Any,
                   quantize_node_activation_fn) -> Tuple[Union[List,torch.Tensor], Union[List,torch.Tensor]]:
    """
    Applying the layer (op_func) to the input tensors (input_tensors).
    If quantized is set to True, and the layer's corresponding node (n) has quantization
    attributes, an additional fake-quantization node is built and appended to the layer.

    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of Pytorch tensors that are the layer's inputs.
        op_func: Module/functional to apply to the input tensors.
        quantize_node_activation_fn: quantization function
    Returns:
        A tuple of Pytorch tensors. The Module/functional output tensors after applying the
        Module/functional to the input tensors.
    """

    op_call_args = n.op_call_args if isinstance(n, FunctionalNode) else []
    functional_kwargs = n.op_call_kwargs if isinstance(n, FunctionalNode) else {}
    if isinstance(n, FunctionalNode) and n.inputs_as_list:
        out_tensors_of_n_float = op_func(input_tensors, *op_call_args, **functional_kwargs)
    else:
        out_tensors_of_n_float = op_func(*input_tensors + op_call_args, **functional_kwargs)

    # Add a fake quant node if the node has an activation threshold.
    out_tensors_of_n = out_tensors_of_n_float
    if n.is_activation_quantization_enabled():
        if isinstance(out_tensors_of_n_float, list):
            out_tensors_of_n_float = torch.cat(out_tensors_of_n_float, dim=0)
        out_tensors_of_n = quantize_node_activation_fn(n, out_tensors_of_n_float)

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
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                 return_float_outputs: bool = False):
        """
        Construct a Pytorch model.

        Args:
            graph: Graph to build its corresponding Pytorch model.
            append2output: List of nodes or OutTensor objects.
            fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
            return_float_outputs: Whether the model returns float tensors or not.
        """
        super(PytorchModel, self).__init__()
        self.graph = graph
        self.node_sort = list(topological_sort(graph))
        self.nodes_dict = {}
        self.append2output = append2output
        self.return_float_outputs = return_float_outputs
        self.fw_info = fw_info
        self._add_modules()

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
        raise NotImplemented(f'{self.__class__.__name__} have to implement a method for quantization activation nodes.')

    def _add_modules(self):
        for n in self.node_sort:
            if not isinstance(n, FunctionalNode):
                if n.type == BufferHolder:
                    self.add_module(n.name, node_builder(n))
                    self.get_submodule(n.name). \
                        register_buffer(n.name, torch.Tensor(n.get_weights_by_keys(BUFFER)).to(get_working_device()))
                else:
                    self.add_module(n.name, node_builder(n))

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
        configurable_nodes = self.graph.get_configurable_sorted_nodes_names()
        for n in self.node_sort:
            input_tensors = _build_input_tensors_list(n,
                                                      self.graph,
                                                      args,
                                                      node_to_output_tensors_dict)

            op_func = self._get_op_func(n, configurable_nodes)

            # Run node operation and fetch outputs
            out_tensors_of_n, out_tensors_of_n_float = _run_operation(n,
                                                                      input_tensors,
                                                                      op_func=op_func,
                                                                      quantize_node_activation_fn=self._quantize_node_activations)

            if isinstance(out_tensors_of_n, list):
                node_to_output_tensors_dict.update({n: out_tensors_of_n})
                node_to_output_tensors_dict_float.update({n: out_tensors_of_n_float})
            else:
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})
                node_to_output_tensors_dict_float.update({n: [out_tensors_of_n_float]})


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
        return node.type if isinstance(node, FunctionalNode) else getattr(self, node.name)


class PyTorchModelBuilder(BaseModelBuilder):
    """
    Builder of PyTorch models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

    def build_model(self) -> Tuple[PytorchModel, UserInformation]:
        """
        Build a PyTorch model and return it.
        Returns: Pytorch model and user information.

        """
        return PytorchModel(self.graph,
                            self.append2output,
                            return_float_outputs=self.return_float_outputs), self.graph.user_info
