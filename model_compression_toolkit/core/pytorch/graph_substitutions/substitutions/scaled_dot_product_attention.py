# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import torch.nn as nn
import torch
import math
from copy import copy
import numpy as np
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common import BaseSubstitution
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.pytorch.constants import DIM
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device


class ScaledDotProductDecomposition(BaseSubstitution):
    """
    Decompose torch.nn.scale_dot_product into its base operators:
        Transpose (over k)
        MatMul(over q and transposed k)
        Mul (for scaling)
        Add (for masking. optional operation, used in cases that attn_mask ig given)
        Dropout
        Softmax
        Matmul.
    """

    def __init__(self):
        """
        Matches scaled_dot_product_attention node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(nn.functional.scaled_dot_product_attention))

    def _get_input_by_name(self, attention_node: FunctionalNode, input_name: str,
                           input_index: int, default_value: any) -> any:
        """
        Search for attention_node input value in op_call_kwargs (using input_name) and op_call_args (using input_index).
        In case the input is not given, returns its default_value.

        """
        if input_name in attention_node.op_call_kwargs:
            return attention_node.op_call_kwargs[input_name]
        elif len(attention_node.op_call_args) > input_index:  # input order: [attn_mask, dropout_p, is_causal]
            return attention_node.op_call_args[input_index]
        return default_value

    def _get_attention_input_nodes(self, graph: Graph, attention_node: FunctionalNode) -> dict:
        q, k, v = 0, 1, 2
        prev_nodes = graph.get_prev_nodes(attention_node, sink_index_sorted=True)
        q_node, k_node, v_node = prev_nodes[q], prev_nodes[k], prev_nodes[v]
        return {"q": q_node, "k": k_node, "v": v_node}

    def _get_transpose_k_node(self, attention_node_name: str, key_node: BaseNode) -> BaseNode:
        input_shape, output_shape = copy(key_node.output_shape[0]), copy(key_node.output_shape[0])
        output_shape[-2], output_shape[-1] = input_shape[-1], input_shape[-2]
        transpose_node = FunctionalNode(name=f"{attention_node_name}_{key_node.name}_transpose",
                                        framework_attr={},
                                        input_shape=input_shape,
                                        output_shape=output_shape,
                                        weights={},
                                        layer_class=torch.transpose,
                                        op_call_args=[-1, -2],  # axes to transpose
                                        op_call_kwargs={},
                                        functional_op=torch.transpose)
        return transpose_node

    def _get_scale_node(self, attention_node: FunctionalNode, q_node: BaseNode, matmul_node: BaseNode) -> FunctionalNode:
        """
        :return: multiplication node that represents multiplication by the scale factor
        """
        scale_name = f'{attention_node.name}_scale'
        q_embd_axis = -1
        input_scale = self._get_input_by_name(attention_node, "scale", 3, None)
        scale_factor = input_scale if input_scale else (1 / math.sqrt(q_node.output_shape[0][q_embd_axis]))
        scale_node = FunctionalNode(name=scale_name,
                                    framework_attr={},
                                    input_shape=(matmul_node.output_shape),
                                    output_shape=matmul_node.output_shape,
                                    weights={},
                                    layer_class=torch.mul,
                                    op_call_args=[scale_factor],
                                    op_call_kwargs={},
                                    functional_op=torch.mul)
        return scale_node

    def _get_matmul_node(self, attention_node_name: str, q_node: BaseNode, transposed_k_node: BaseNode) -> BaseNode:
        matmul1_output_shape = copy(q_node.output_shape[0])
        matmul1_output_shape[-2] = q_node.output_shape[0][-2]
        matmul1_output_shape[-1] = transposed_k_node.output_shape[-1]
        matmul_name = f'{attention_node_name}_matmul1'
        return FunctionalNode(name=matmul_name,
                              framework_attr={},
                              input_shape=(tuple(q_node.output_shape[0]), tuple(transposed_k_node.output_shape)),
                              output_shape=tuple(matmul1_output_shape),
                              weights={},
                              layer_class=torch.matmul,
                              op_call_args=[],
                              op_call_kwargs={},
                              functional_op=torch.matmul)

    def _get_mask_node(self, attention_node: FunctionalNode, scale_node: FunctionalNode) -> FunctionalNode:
        """
        :return: Add operator node with the mask tensor as input. In case there is no mask tensor, returns None.
        """
        attention_mask_tensor = self._get_attention_mask_tensor(attention_node)
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

    def _get_softmax_node(self, attention_node_name: str, in_out_shape: tuple) -> BaseNode:
        softmax_name = f'{attention_node_name}_softmax'
        return BaseNode(name=softmax_name,
                        framework_attr={DIM: -1},
                        input_shape=in_out_shape,
                        output_shape=in_out_shape,
                        weights={},
                        layer_class=nn.Softmax)

    def _get_matmul2_node(self, attention_node_name: str, softmax_node: BaseNode, v_node: BaseNode) -> FunctionalNode:
        matmul2_output_shape = list(copy(softmax_node.output_shape))
        matmul2_output_shape[-2] = softmax_node.output_shape[-2]
        matmul2_output_shape[-1] = v_node.output_shape[0][-1]
        matmul2_name = f'{attention_node_name}_matmul2'
        return FunctionalNode(name=matmul2_name,
                              framework_attr={},
                              input_shape=(tuple(softmax_node.output_shape), tuple(v_node.output_shape[0])),
                              output_shape=tuple(matmul2_output_shape),
                              weights={},
                              layer_class=torch.matmul,
                              op_call_args=[],
                              op_call_kwargs={},
                              functional_op=torch.matmul)

    def _get_attention_mask_tensor(self, attention_node: FunctionalNode) -> torch.Tensor:
        """
        :return: mask tensor given as part of attention node input.
        Since MCT doesn't support infinite values, we don't support is_causal (torch.nn.scale_dot_product_attention
        argument) and boolean mask tensor, as they both require -inf values.
        """
        device = get_working_device()
        is_causal = self._get_input_by_name(attention_node, "is_causal", 2, False)
        if is_causal:
            raise NotImplementedError("scaled_dot_product_attention is_causal feature is not implemented.")
        input_weights = list(attention_node.weights.values())
        attn_mask = input_weights[0] if len(input_weights) > 0 else None
        if attn_mask is not None and (attn_mask.dtype == "bool"):
            raise NotImplementedError(
                "scaled_dot_product_attention attn_mask is of type boolean, which is not supported.")
        if attn_mask is not None and (not np.isfinite(attn_mask).all()):
            raise NotImplementedError(
                "scaled_dot_product_attention attn_mask contains infinite value, which is not supported.")
        return torch.from_numpy(attn_mask).to(device) if attn_mask is not None else None

    def _get_dropout_node(self, attention_node: FunctionalNode, in_out_shape: tuple) -> BaseNode:
        dropout_p = attention_node.op_call_kwargs.get('dropout_p', 0)
        dropout_name = f'{attention_node.name}_dropout'
        return BaseNode(name=dropout_name,
                        framework_attr={"p": dropout_p},
                        input_shape=in_out_shape,
                        output_shape=in_out_shape,
                        weights={},
                        layer_class=nn.Dropout)

    def substitute(self, graph: Graph, attention_node: FunctionalNode) -> Graph:
        """
        Removes a scaled_dot_product_attention node from the graph, and replaces it with a compatible graph that
        consists of:
            Transpose (over k)
            MatMul(over q and transposed k)
            Mul (for scaling)
            Add (for masking. optional operation, used in cases that attn_mask ig given)
            Dropout
            Softmax
            Matmul.
        :param graph: A Graph to apply substitution on
        :param attention_node: the node to replace
        :return: A graph after the substitution
        """
        print("In scale_dot_product_attention substitution@@@@@@@@")
        input_nodes = self._get_attention_input_nodes(graph, attention_node)
        q_node, k_node, v_node = input_nodes["q"], input_nodes["k"], input_nodes["v"]
        transpose_k_node = self._get_transpose_k_node(attention_node.name, k_node)
        matmul_node = self._get_matmul_node(attention_node.name, q_node, transpose_k_node)
        scale_node = self._get_scale_node(attention_node, q_node, matmul_node)
        mask_node = self._get_mask_node(attention_node, scale_node)
        softmax_node = self._get_softmax_node(attention_node.name, matmul_node.output_shape)
        dropout_node = self._get_dropout_node(attention_node, softmax_node.output_shape)
        matmul2_node = self._get_matmul2_node(attention_node.name, softmax_node, v_node)

        graph.add_node_with_in_edges(transpose_k_node, [k_node])
        graph.add_node_with_in_edges(matmul_node, [q_node, transpose_k_node])
        graph.add_node_with_in_edges(scale_node, [matmul_node])
        if mask_node:
            graph.add_node_with_in_edges(mask_node, [scale_node])
        graph.add_node_with_in_edges(softmax_node, [mask_node if mask_node else scale_node])
        graph.add_node_with_in_edges(dropout_node, [softmax_node])
        graph.add_node_with_in_edges(matmul2_node, [dropout_node if dropout_node else softmax_node, v_node])

        graph_outputs = graph.get_outputs()
        for i, g_out in enumerate(graph_outputs):
            if g_out.node == attention_node:
                graph_outputs[i] = OutTensor(node=matmul2_node, node_out_index=g_out.node_out_index)

        graph.reconnect_out_edges(current_node=attention_node, new_node=matmul2_node)
        graph.remove_edge(q_node, attention_node)
        graph.remove_edge(k_node, attention_node)
        graph.remove_edge(v_node, attention_node)
        graph.remove_node(attention_node, new_graph_outputs=graph_outputs)
        return graph
