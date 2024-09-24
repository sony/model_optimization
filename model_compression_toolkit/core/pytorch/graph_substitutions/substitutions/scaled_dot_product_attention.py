import torch.nn as nn
import torch
import math
from copy import copy
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common import BaseSubstitution
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.pytorch.constants import DIM
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device


class ScaledDotProductDecomposition(BaseSubstitution):
    """
    Removes a MultiHeadAttention node from the graph,
    and replaces it with a compatible graph that consists of Conv, MatMul, Softmax and stacked layers.
    """

    def __init__(self):
        """
        Matches MultiHeadAttention node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(nn.functional.scaled_dot_product_attention))
        self.device = get_working_device()

    def _get_attention_input_nodes(self, graph: Graph, attention_node: BaseNode) -> dict:
        q, k, v = 0, 1, 2
        prev_nodes = graph.get_prev_nodes(attention_node)
        q_node, k_node, v_node = prev_nodes[q], prev_nodes[k], prev_nodes[v]
        assert q_node.name == "q" and k_node.name == "k" and v_node.name == "v", \
            "Bad assumption on attention input nodes order"
        return {"q": q_node, "k": k_node, "v": v_node}

    def _get_transpose_node(self, attention_node: BaseNode, node: BaseNode) -> BaseNode:
        input_shape, output_shape = copy(node.output_shape[0]), copy(node.output_shape[0])
        output_shape[-2], output_shape[-1] = input_shape[-1], input_shape[-2]
        transpose_node = FunctionalNode(name=f"{attention_node.name}_{node.name}_transpose",
                                          framework_attr={},
                                          input_shape=input_shape,
                                          output_shape=output_shape,
                                          weights={},
                                          layer_class=torch.transpose,
                                          op_call_args=[1, 2],
                                          op_call_kwargs={},
                                          functional_op=torch.transpose)
        return transpose_node

    def _get_scale_node(self, attention_node, q_node, matmul_node):
        scale_name = f'{attention_node.name}_scale'
        q_embd_axis = -1
        scale_factor = math.sqrt(q_node.output_shape[0][q_embd_axis])  # todo: validate the dimention is correct
        scale_node = FunctionalNode(name=scale_name,
                                     framework_attr={},
                                     input_shape=(matmul_node.output_shape),
                                     output_shape=matmul_node.output_shape,
                                     weights={},
                                     layer_class=torch.div,
                                     op_call_args=[scale_factor],
                                     op_call_kwargs={},
                                     functional_op=torch.div)
        return scale_node

    def _get_matmul_node(self, attention_node: BaseNode, q_node: BaseNode, k_node: BaseNode) -> BaseNode:
        q_batch_axis = 0
        q_and_k_embd_axis = -1  # d_k == d
        k_seq_axis = -2
        q_seq_axis = -2

        matmul1_output_shape = copy(q_node.output_shape[0])
        matmul1_output_shape[-2] = q_node.output_shape[0][q_seq_axis]
        matmul1_output_shape[-1] = k_node.output_shape[-1]

        matmul_name = f'{attention_node.name}_matmul1'
        matmul_node = FunctionalNode(name=matmul_name,
                                     framework_attr={},
                                     input_shape=(tuple(q_node.output_shape[0]), tuple(k_node.output_shape)),
                                     output_shape=tuple(matmul1_output_shape),
                                     weights={},
                                     layer_class=torch.matmul,
                                     op_call_args=[],
                                     op_call_kwargs={},
                                     functional_op=torch.matmul)
        return matmul_node

    def _get_mask_node(self, attention_node, q_node, k_node, scale_node):
        attention_mask_tensor = self._get_attention_mask_tensor(attention_node, q_node, k_node)
        if attention_mask_tensor is None:
            return None
        mask_node_name = f'{attention_node.name}_mask'
        return FunctionalNode(name=mask_node_name,
                              framework_attr={},
                              input_shape=(scale_node.output_shape),
                              output_shape=scale_node.output_shape,
                              weights={},
                              layer_class=torch.add,
                              op_call_args=[],
                              op_call_kwargs={'other': attention_mask_tensor},
                              functional_op=torch.add)

    def _get_softmax_node(self, attention_node, in_out_shape):
        softmax_name = f'{attention_node.name}_softmax'
        return BaseNode(name=softmax_name,
                        framework_attr={DIM: -1},
                        input_shape=in_out_shape,
                        output_shape=in_out_shape,
                        weights={},
                        layer_class=nn.Softmax)

    def _get_matmul2_node(self, attention_node, softmax_node, v_node):
        matmul2_output_shape = list(copy(softmax_node.output_shape))
        matmul2_output_shape[-2] = softmax_node.output_shape[-2]
        matmul2_output_shape[-1] = v_node.output_shape[0][-1]
        matmul2_name = f'{attention_node.name}_matmul2'
        return FunctionalNode(name=matmul2_name,
                            framework_attr={},
                            input_shape=(tuple(softmax_node.output_shape), tuple(v_node.output_shape[0])),
                            output_shape=tuple(matmul2_output_shape),
                            weights={},
                            layer_class=torch.matmul,
                            op_call_args=[],
                            op_call_kwargs={},
                            functional_op=torch.matmul)

    def _get_attention_mask_tensor(self, attention_node, q_node, k_node):
        target_seq_len = q_node.output_shape[0][-2]
        source_seq_len = k_node.output_shape[0][-2]
        attn_bias = torch.zeros(target_seq_len, source_seq_len).to(self.device) # todo: check if need to add dtype=query.dtype
        # attn_mask = attention_node.op_call_kwargs['attn_mask']
        is_causal = attention_node.op_call_kwargs.get('is_causal', False)
        if is_causal:
            raise NotImplementedError("scaled_dot_product_attention is_causal feature is not implemented.")
        input_weights = list(attention_node.weights.values())
        attn_mask = input_weights[0] if len(input_weights) > 0 else None
        # if is_causal:
        #     assert attn_mask is None, "In case 'is_causal' is set to True, 'attn_mask' must be None"
        #     temp_mask = torch.ones(target_seq_len, source_seq_len, dtype=torch.bool).tril(diagonal=0).to(self.device)
        #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf")).to(self.device)
        #     attn_bias.to(torch.float32)
        #
        # if attn_mask is not None:
        #     if attn_mask.dtype == torch.bool:
        #         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        #     else:
        # attn_bias += torch.from_numpy(attn_mask).to(self.device) # todo: ask why attn_mask have changed to numpy ndarray
        return attn_mask

    def _get_dropout_node(self, attention_node, in_out_shape):
        dropout_p = attention_node.op_call_kwargs['dropout_p']
        if dropout_p == 0:
            return None
        dropout_name = f'{attention_node.name}_dropout'
        return BaseNode(name=dropout_name,
                                framework_attr={"p": attention_node.op_call_kwargs['dropout_p']},
                                input_shape=in_out_shape,
                                output_shape=in_out_shape,
                                weights={},
                                layer_class=nn.Dropout)

    def substitute(self,
                   graph: Graph,
                   attention_node: BaseNode) -> Graph:

        input_nodes = self._get_attention_input_nodes(graph, attention_node)
        q_node, k_node, v_node = input_nodes["q"], input_nodes["k"], input_nodes["v"]
        transpose_k_node = self._get_transpose_node(attention_node, k_node)
        matmul_node = self._get_matmul_node(attention_node, q_node, transpose_k_node)
        scale_node = self._get_scale_node(attention_node, q_node, matmul_node)
        mask_node = self._get_mask_node(attention_node, q_node, k_node, scale_node)
        softmax_node = self._get_softmax_node(attention_node, matmul_node.output_shape)
        dropout_node = self._get_dropout_node(attention_node, softmax_node.output_shape)
        matmul2_node = self._get_matmul2_node(attention_node, softmax_node, v_node)

        graph.add_node_with_in_edges(transpose_k_node, [k_node])
        graph.add_node_with_in_edges(matmul_node, [q_node, transpose_k_node])
        graph.add_node_with_in_edges(scale_node, [matmul_node])
        if mask_node:
            graph.add_node_with_in_edges(mask_node, [scale_node])
        graph.add_node_with_in_edges(softmax_node, [mask_node if mask_node else scale_node])
        if dropout_node:
            graph.add_node_with_in_edges(dropout_node, [softmax_node])
        graph.add_node_with_in_edges(matmul2_node, [dropout_node if dropout_node else softmax_node, v_node])

        graph.remove_edge(q_node, attention_node)
        graph.remove_edge(k_node, attention_node)
        graph.remove_edge(v_node, attention_node)
        graph.remove_node(attention_node, new_graph_outputs=[OutTensor(matmul2_node, 0)])
        return graph
