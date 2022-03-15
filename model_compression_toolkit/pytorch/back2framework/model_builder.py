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
import inspect
from typing import Tuple, Any, Dict, List

import numpy as np
import torch
from networkx import topological_sort

from model_compression_toolkit import common, FrameworkInfo
from model_compression_toolkit.common import BaseNode, Graph
from model_compression_toolkit.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.pytorch.reader.graph_builders import DummyPlaceHolder
from model_compression_toolkit.pytorch.mixed_precision.mixed_precision_wrapper import PytorchMixedPrecisionWrapper


def build_input_tensors_list(node: BaseNode,
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


def run_operation(n: BaseNode,
                  input_tensors: List,
                  op_func: Any,
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED) -> List[torch.Tensor]:
    """
    Applying the layer (op_func) to the input tensors (input_tensors).
    If quantized is set to True, and the layer's corresponding node (n) has quantization
    attributes, an additional fake-quantization node is built and appended to the layer.
    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of Pytorch tensors that are the layer's inputs.
        op_func: Module/functional to apply to the input tensors.
        mode: model quantiztion mode from ModelBuilderMode
    Returns:
        A list of Pytorch tensors. The Module/functional output tensors after applying the
        Module/functional to the input tensors.
    """

    op_call_args = n.op_call_args if isinstance(n, FunctionalNode) else []
    functional_kwargs = n.op_call_kwargs if isinstance(n, FunctionalNode) else {}
    if isinstance(n, FunctionalNode) and n.inputs_as_list:
        out_tensors_of_n = op_func(input_tensors, *op_call_args, **functional_kwargs)
    else:
        out_tensors_of_n = op_func(*input_tensors + op_call_args, **functional_kwargs)

    # Add a fake quant node if the node has an activation threshold.
    if mode == ModelBuilderMode.QUANTIZED and n.is_activation_quantization_enabled() \
            and n.final_activation_quantization_cfg:
        out_tensors_of_n = n.final_activation_quantization_cfg.quantize_node_output(out_tensors_of_n)

    return out_tensors_of_n


def generate_outputs(
        out_nodes: List[BaseNode],
        node_to_output_tensors_dict: dict):
    """
    Args:
        out_nodes: List of output nodes.
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.
    Returns: List of output tensor/s for the model
    """
    output = []
    for n in out_nodes:
        out_tensors_of_n = node_to_output_tensors_dict.get(n)
        if len(out_tensors_of_n) > 1:
            output.append(out_tensors_of_n)
        else:
            output += out_tensors_of_n
    return output


class PytorchModelBuilder(torch.nn.Module):
    """
    Class for reconstructing a Pytorch model from a graph
    """
    def __init__(self, graph: Graph,
                 mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                 append2output: List[Any] = None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO):
        """
        Construct a Pytorch model.
        Args:
            graph: Graph to build its corresponding Pytorch model.
            mode: Building mode. Read ModelBuilderMode description for more info.
            append2output: List of nodes or OutTensor objects.
            fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        """
        super(PytorchModelBuilder, self).__init__()
        self.graph = graph
        self.mode = mode
        self.node_sort = list(topological_sort(graph))
        self.nodes_dict = {}
        self.append2output = append2output

        if mode == ModelBuilderMode.MIXEDPRECISION:
            configurable_nodes = self.graph.get_configurable_sorted_nodes()
            for n in self.node_sort:
                if not isinstance(n, FunctionalNode):
                    if n in configurable_nodes:
                        self.add_module(n.name, PytorchMixedPrecisionWrapper(n, fw_info))
                    else:
                        self.add_module(n.name, node_builder(n))
        else:
            for n in self.node_sort:
                if not isinstance(n, FunctionalNode):
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
        for n in self.node_sort:
            input_tensors = build_input_tensors_list(n,
                                                     self.graph,
                                                     args,
                                                     node_to_output_tensors_dict)
            out_tensors_of_n = run_operation(n,  # Run node operation and fetch outputs
                                             input_tensors,
                                             op_func=n.type if isinstance(n, FunctionalNode) else getattr(self, n.name),
                                             mode=self.mode)
            if isinstance(out_tensors_of_n, list):
                node_to_output_tensors_dict.update({n: out_tensors_of_n})
            else:
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

        if self.append2output:
            outputs = generate_outputs(self.append2output,
                                       node_to_output_tensors_dict)
        else:
            outputs = generate_outputs([ot.node for ot in self.graph.get_outputs()],
                                       node_to_output_tensors_dict)
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs


def model_builder(graph: common.Graph,
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                  append2output: List[Any] = None,
                  fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO) -> Tuple[torch.nn.Module, Any]:
    """
    Build a Pytorch model from a graph representing the model.
    The model is built by converting the graph nodes to torch modules and initializing a PytorchModelBuilder class.
    When the model is not built in float mode, the graph is transformed by additional substitutions.
    Args:
        graph: Graph to build its corresponding Pytorch model.
        mode: Building mode. Read ModelBuilderMode description for more info.
        append2output: List of nodes or OutTensor objects. In float building mode,
        when the list contains nodes, all output tensors of all nodes are set as the model outputs.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
    Returns:
        A tuple of the model, and an UserInformation object.
    """

    model = PytorchModelBuilder(graph, mode, append2output, fw_info)

    return model, graph.user_info
