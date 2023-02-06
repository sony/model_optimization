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
from typing import Any, Dict, List

import torch
import torch.autograd as autograd
from networkx import topological_sort
from tqdm import tqdm
import numpy as np

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.constants import EPS, MIN_JACOBIANS_ITER, JACOBIANS_COMP_TOLERANCE
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy
from model_compression_toolkit.core.common.logger import Logger


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
        if len(out_tensors_of_n) > 1 or isinstance(out_tensors_of_n[0], tuple):
            if isinstance(out_tensors_of_n[0], tuple):
                out_tensors_of_n = out_tensors_of_n[0]
            out_tensors_of_n = [torch.cat(out_tensors_of_n)]
            output.append(torch.concat(out_tensors_of_n))
        else:
            output += out_tensors_of_n
    return output


class PytorchModelGradients(torch.nn.Module):
    """
    A Pytorch Module class for producing differentiable outputs from a given model's graph representation.
    """
    def __init__(self,
                 graph_float: common.Graph,
                 interest_points: List[BaseNode],
                 output_list: List[BaseNode]):
        """
        Construct a PytorchModelGradients model.

        Args:
            graph_float: Model's Graph representation to evaluate the outputs according to.
            interest_points: List of nodes in the graph which we want to produce outputs for.
            output_list: List of nodes that considered as model's output for the purpose of gradients computation.
        """

        super(PytorchModelGradients, self).__init__()
        self.graph_float = graph_float
        self.node_sort = list(topological_sort(graph_float))
        self.interest_points = interest_points
        self.output_list = output_list
        self.interest_points_tensors = []

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

            if isinstance(out_tensors_of_n, list) or isinstance(out_tensors_of_n, tuple):
                output_t = []
                for t in out_tensors_of_n:
                    if n in self.interest_points:
                        if t.requires_grad:
                            t.retain_grad()
                            self.interest_points_tensors.append(t)
                        else:
                            # We get here in case we have an output node, which is an interest point,
                            # but it is not differentiable. We need to add this dummy tensor in order to include this
                            # node in the future weights computation.
                            # Note that this call is excluded from tests coverage,
                            # since we do not suppose to get here - there is no valid operation that is both
                            # non-differentiable and return output as a list or a tuple
                            self.interest_points_tensors.append(torch.tensor([0.0],  # pragma: no cover
                                                                             requires_grad=True,
                                                                             device=t.device))
                            break  # pragma: no cover
                    output_t.append(t)
                if isinstance(out_tensors_of_n, tuple):
                    # If the node's output is a Tuple, then we want to keep it as a Tuple
                    output_t = [tuple(output_t)]
                node_to_output_tensors_dict.update({n: output_t})
            else:
                assert isinstance(out_tensors_of_n, torch.Tensor)
                if n in self.interest_points:
                    if out_tensors_of_n.requires_grad:
                        out_tensors_of_n.retain_grad()
                        self.interest_points_tensors.append(out_tensors_of_n)
                    else:
                        # We get here in case we have an output node, which is an interest point,
                        # but it is not differentiable. We need to add this dummy tensor in order to include this
                        # node in the future weights computation.
                        self.interest_points_tensors.append(torch.tensor([0.0],
                                                                         requires_grad=True,
                                                                         device=out_tensors_of_n.device))
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

        outputs = generate_outputs(self.output_list,
                                   node_to_output_tensors_dict)

        return outputs


def pytorch_iterative_approx_jacobian_trace(graph_float: common.Graph,
                                            model_input_tensors: Dict[BaseNode, torch.Tensor],
                                            interest_points: List[BaseNode],
                                            output_list: List[BaseNode],
                                            all_outputs_indices: List[int],
                                            alpha: float = 0.3,
                                            n_iter: int = 50,
                                            norm_weights: bool = True) -> List[float]:
    """
    Computes an approximation of the power of the Jacobian trace of a Pytorch model's outputs with respect to the feature maps of
    the set of given interest points. It then uses the power of the Jacobian trace for each interest point and normalized the
    values, to be used as weights for weighted average in distance metric computation.

    Args:
        graph_float: Graph to build its corresponding Pytorch model.
        model_input_tensors: A mapping between model input nodes to an input batch torch Tensor.
        interest_points: List of nodes which we want to get their feature map as output, to calculate distance metric.
        output_list: List of nodes that considered as model's output for the purpose of gradients computation.
        all_outputs_indices: Indices of the model outputs and outputs replacements (if exists),
            in a topological sorted interest points list.
        alpha: A tuning parameter to allow calibration between the contribution of the output feature maps returned
            weights and the other feature maps weights (since the gradient of the output layers does not provide a
            compatible weight for the distance metric computation).
        n_iter: The number of random iterations to calculate the approximated power of the Jacobian trace for each interest point.
        norm_weights: Whether to normalize the returned weights (to get values between 0 and 1).

    Returns: A list of (possibly normalized) jacobian-based weights to be considered as the relevancy that each interest
    point's output has on the model's output.
    """

    # Set inputs to require_grad
    for n, input_tensor in model_input_tensors.items():
        input_tensor.requires_grad_()

    model_grads_net = PytorchModelGradients(graph_float=graph_float,
                                            interest_points=interest_points,
                                            output_list=output_list)

    # Run model inference
    output_tensors = model_grads_net(model_input_tensors)
    device = output_tensors[0].device


    # Concat outputs
    # First, we need to unfold all outputs that are given as list, to extract the actual output tensors
    unfold_outputs = []
    for output in output_tensors:
        if isinstance(output, List):
            unfold_outputs += output
        else:
            unfold_outputs.append(output)

    r_outputs = [torch.reshape(output, shape=[output.shape[0], -1]) for output in unfold_outputs]

    concat_axis_dim = [o.shape[0] for o in r_outputs]
    if not all(d == concat_axis_dim[0] for d in concat_axis_dim):
        Logger.critical("Can't concat model's outputs for gradients calculation since the shape of the first axis "  # pragma: no cover
                        "is not equal in all outputs.")

    output = torch.concat(r_outputs, dim=1)

    ipts_jac_trace_approx = []
    for ipt in tqdm(model_grads_net.interest_points_tensors):  # Per Interest point activation tensor
        trace_jv = []
        for j in range(n_iter):  # Approximation iterations
            # Getting a random vector with normal distribution
            v = torch.randn(output.shape, device=device)
            f_v = torch.sum(v * output)

            # Computing the jacobian approximation by getting the gradient of (output * v)
            jac_v = autograd.grad(outputs=f_v,
                                  inputs=ipt,
                                  retain_graph=True,
                                  allow_unused=True)[0]
            if jac_v is None:
                # In case we have an output node, which is an interest point, but it is not differentiable,
                # we still want to set some weight for it. For this, we need to add this dummy tensor to the ipt
                # jacobian traces list.
                trace_jv.append(torch.tensor([0.0],
                                             requires_grad=True,
                                             device=device))
                break
            jac_v = torch.reshape(jac_v, [jac_v.shape[0], -1])
            jac_trace_approx = torch.mean(torch.sum(torch.pow(jac_v, 2.0)))

            # If the change to the mean Jacobian approximation is insignificant we stop the calculation
            if j > MIN_JACOBIANS_ITER:
                delta = torch.mean(torch.stack([jac_trace_approx, *trace_jv])) - torch.mean(
                    torch.stack(trace_jv))
                if torch.abs(delta) / (torch.abs(torch.mean(torch.stack(trace_jv))) + 1e-6) < JACOBIANS_COMP_TOLERANCE:
                    trace_jv.append(jac_trace_approx)
                    break

            trace_jv.append(jac_trace_approx)
        ipts_jac_trace_approx.append(2*torch.mean(torch.stack(trace_jv))/output.shape[-1])  # Get averaged jacobian trace approximation

    ipts_jac_trace_approx = torch_tensor_to_numpy(torch.Tensor(ipts_jac_trace_approx))  # Just to get one tensor instead of list of tensors with single element

    if norm_weights:
        return _normalize_weights(ipts_jac_trace_approx, all_outputs_indices, alpha)
    else:
        return ipts_jac_trace_approx


def _normalize_weights(jacobians_traces: np.ndarray,
                       all_outputs_indices: List[int],
                       alpha: float) -> List[float]:
    """
    Output layers or layers that come after the model's considered output layers,
    are assigned with a constant normalized value, according to the given alpha variable and the number of such layers.
    Other layers returned weights are normalized by dividing the jacobian-based weights value by the sum of all other values.

    Args:
        jacobians_traces: The approximated average jacobian-based weights of each interest point.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: Normalized list of jacobian-based weights (for each interest point).

    """

    jacobians_without_outputs = [jacobians_traces[i] for i in range(len(jacobians_traces))
                                 if i not in all_outputs_indices]
    sum_without_outputs = sum(jacobians_without_outputs)

    normalized_grads_weights = [_get_normalized_weight(grad, i, sum_without_outputs, all_outputs_indices, alpha)
                                for i, grad in enumerate(jacobians_traces)]

    return normalized_grads_weights


def _get_normalized_weight(grad: torch.Tensor,
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

    Returns: A normalized jacobian-based weights.

    """

    if i in all_outputs_indices:
        return alpha / len(all_outputs_indices)
    else:
        return ((1 - alpha) * grad / (sum_without_outputs + EPS)).item()
