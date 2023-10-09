from typing import List, Any, Dict

import torch
from networkx import topological_sort

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.hessian import TraceHessianRequest
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.constants import BUFFER
from model_compression_toolkit.core.pytorch.reader.node_holders import BufferHolder, DummyPlaceHolder
from model_compression_toolkit.core.pytorch.utils import get_working_device


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
        input_idx = graph.get_inputs().index(node)
        input_tensors = [inputs[input_idx]]
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
                 graph_float: Graph,
                 trace_hessian_request: TraceHessianRequest
                 ):
        """
        Construct a PytorchModelGradients model.

        Args:
            graph_float: Model's Graph representation to evaluate the outputs according to.
            trace_hessian_request: TraceHessianRequest to build the model according to.
        """

        super(PytorchModelGradients, self).__init__()
        self.graph_float = graph_float
        self.trace_hessian_request = trace_hessian_request
        self.node_sort = list(topological_sort(graph_float))
        self.output_list = [o.node for o in graph_float.output_nodes]
        self.interest_points_tensors = []

        for n in self.node_sort:
            if not isinstance(n, FunctionalNode):
                if n.type == BufferHolder:
                    self.add_module(n.name, node_builder(n))
                    self.get_submodule(n.name). \
                        register_buffer(n.name,
                                        torch.Tensor(n.get_weights_by_keys(BUFFER)).to(get_working_device()))
                else:
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
                    if n.name == self.trace_hessian_request.target_node.name:
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
                if n.name == self.trace_hessian_request.target_node.name:
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