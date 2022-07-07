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
from typing import Any, Dict, List

import torch
from networkx import topological_sort

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.constants import EPS
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.reader.graph_builders import DummyPlaceHolder


def build_input_tensors_list(node: BaseNode,
                             graph: Graph,
                             inputs: Dict[BaseNode, List],
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
        input_tensors = [inputs[node]]
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
                  op_func: Any) -> List[torch.Tensor]:
    """
    Applying the layer (op_func) to the input tensors (input_tensors).
    If quantized is set to True, and the layer's corresponding node (n) has quantization
    attributes, an additional fake-quantization node is built and appended to the layer.
    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of Pytorch tensors that are the layer's inputs.
        op_func: Module/functional to apply to the input tensors.
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


class PytorchModelGradients(torch.nn.Module):
    """
    A Pytorch Module class for producing differentiable outputs from a given model's graph representation.
    """
    def __init__(self,
                 graph_float: common.Graph,
                 interest_points: List[BaseNode]):
        """
        Construct a PytorchModelGradients model.

        Args:
            graph_float: Model's Graph representation to evaluate the outputs according to.
            interest_points: List of nodes in the graph which we want to produce outputs for.
        """

        super(PytorchModelGradients, self).__init__()
        self.graph_float = graph_float
        self.node_sort = list(topological_sort(graph_float))
        self.interest_points = interest_points

        for n in self.node_sort:
            if not isinstance(n, FunctionalNode):
                self.add_module(n.name, node_builder(n))

    def forward(self,
                *args: Any) -> Any:
        """
        Args:
            args: argument input tensors to model, which is a mappings between an input node and its input tensor.

        Returns:
            A list of output tensors for each of the model's pre-defined interest points.
        """

        input_node_to_input_tensor = args[0]
        node_to_output_tensors_dict = dict()
        for n in self.node_sort:
            input_tensors = build_input_tensors_list(n,
                                                     self.graph_float,
                                                     input_node_to_input_tensor,
                                                     node_to_output_tensors_dict)

            op_func = n.type if isinstance(n, FunctionalNode) else getattr(self, n.name)
            out_tensors_of_n = run_operation(n,  # Run node operation and fetch outputs
                                             input_tensors,
                                             op_func=op_func)

            if isinstance(out_tensors_of_n, list):
                output_t = []
                for t in out_tensors_of_n:
                    if n in self.interest_points and t.requires_grad:
                        t.retain_grad()
                    output_t.append(t)
                node_to_output_tensors_dict.update({n: output_t})
            else:
                assert isinstance(out_tensors_of_n, torch.Tensor)
                if n in self.interest_points and out_tensors_of_n.requires_grad:
                    out_tensors_of_n.retain_grad()
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

        outputs = generate_outputs(self.interest_points,
                                   node_to_output_tensors_dict)

        return outputs


def pytorch_model_grad(graph_float: common.Graph,
                       model_input_tensors: Dict[BaseNode, torch.Tensor],
                       interest_points: List[BaseNode],
                       all_outputs_indices: List[int],
                       alpha: float = 0.3) -> List[float]:
    """
    Computes the gradients of a Pytorch model's outputs with respect to the feature maps of the set of given
    interest points. It then uses the gradients to compute the hessian trace for each interest point and normalized the
    values, to be used as weights for weighted average in mixed-precision distance metric computation.

    Args:
        graph_float: Graph to build its corresponding Pytorch model.
        model_input_tensors: A mapping between model input nodes to an input batch torch Tensor.
        interest_points: List of nodes which we want to get their feature map as output, to calculate distance metric.
        all_outputs_indices: Indices of the model outputs and outputs replacements (if exists),
            in a topological sorted interest points list.
        alpha: A tuning parameter to allow calibration between the contribution of the output feature maps returned
            weights and the other feature maps weights (since the gradient of the output layers does not provide a
            compatible weight for the distance metric computation).

    Returns: A list of normalized gradients to be considered as the relevancy that each interest
    point's output has on the model's output.
    """

    for n, input_tensor in model_input_tensors.items():
        input_tensor.requires_grad_()

    model_grads_net = PytorchModelGradients(graph_float=graph_float,
                                            interest_points=interest_points)

    output_tensors = model_grads_net(model_input_tensors)

    ############################################
    # Compute Gradients
    ############################################
    device = output_tensors[0].device
    output_tensors_for_loss = [output_tensors[i] for i in all_outputs_indices]
    output_loss = torch.tensor([0.0], requires_grad=True, device=device)
    for output in output_tensors_for_loss:
        output = torch.reshape(output, shape=(output.shape[0], -1))
        output_loss = torch.add(output_loss, torch.mean(torch.sum(output, dim=-1)))

    output_loss.backward()

    ipt_gradients = [torch.Tensor([0.0]) if t.grad is None else t.grad for t in output_tensors]
    r_ipt_gradients = [torch.reshape(t, shape=(t.shape[0], -1)) for t in ipt_gradients]
    hessian_trace_aprrox = [torch.mean(torch.sum(torch.pow(ipt_grad, 2.0), dim=-1)) for ipt_grad in r_ipt_gradients]

    # Output layers or layers that come after the model's considered output layers,
    # are assigned with a constant normalized value,
    # according to the given alpha variable and the number of such layers.
    # Other layers returned weights are normalized by dividing the hessian value by the sum of all other values.
    hessians_without_outputs = [hessian_trace_aprrox[i] for i in range(len(hessian_trace_aprrox))
                                if i not in all_outputs_indices]
    sum_without_outputs = sum(hessians_without_outputs)
    normalized_grads_weights = [get_normalized_weight(grad,
                                                      i,
                                                      sum_without_outputs,
                                                      all_outputs_indices,
                                                      alpha)
                                for i, grad in enumerate(hessian_trace_aprrox)]

    return normalized_grads_weights


def get_normalized_weight(grad: torch.Tensor,
                          i: int,
                          sum_without_outputs: float,
                          all_outputs_indices: List[int],
                          alpha: float) -> float:
    """
    Normalizes the node's gradient value. If it is an output or output replacement node than the normalized value is
    a constant, otherwise, it is normalized by dividing with the sum of all gradient values.

    Args:
        grad: The gradient value tensor.
        i: The index of the node in the sorted interest points list.
        sum_without_outputs: The sum of all gradients of nodes that are not considered outputs.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: A normalized gradient value.

    """

    if i in all_outputs_indices:
        return alpha / len(all_outputs_indices)
    else:
        return ((1 - alpha) * grad / (sum_without_outputs + EPS)).item()
