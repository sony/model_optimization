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


import numpy as np
import torch
import torch.nn as nn
import operator
from typing import List

from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.pytorch.constants import KERNEL, BIAS, NUM_HEADS, KEY_DIM, VALUE_DIM, \
    EMBED_DIM, BIAS_K, BIAS_V, ADD_ZERO_ATTN, BATCH_FIRST, OUT_PROJ_WEIGHT, OUT_PROJ_BIAS, \
    V_PROJ_WEIGHT, K_PROJ_WEIGHT, Q_PROJ_WEIGHT, IN_PROJ_WEIGHT, IN_PROJ_BIAS, DIM, KERNEL_SIZE, \
    IN_CHANNELS, OUT_CHANNELS


class MHAParams:
    """
    A data class to hold all relevant parameters from the MHA node framework attributes
    """

    def __init__(self,
                 mha_node: BaseNode):
        """
        Extract MHA params from layer attributes
        Args:
            mha_node: MHA node
        """

        # Only batch first network is supported
        if BATCH_FIRST in mha_node.framework_attr.keys():
            if mha_node.framework_attr[BATCH_FIRST] is not True:
                Logger.error('Only batch first network is supported')  # pragma: no cover
        else:
            Logger.error('Only batch first network is supported')  # pragma: no cover

        # Add Zero Attn feature is Not Implemented
        if ADD_ZERO_ATTN in mha_node.framework_attr.keys():
            if mha_node.framework_attr[ADD_ZERO_ATTN] is not False:
                Logger.error('Add Zero Attn feature is Not Implemented')  # pragma: no cover

        # Check if Add Bias KV feature is Active
        if BIAS_K and BIAS_V in mha_node.weights.keys():
            if mha_node.weights[BIAS_K] and mha_node.weights[BIAS_V] is not None:
                Logger.error('Add BIAS_KV feature is Not Implemented')  # pragma: no cover

        self.embed_dim = mha_node.framework_attr[EMBED_DIM]
        self.num_heads = mha_node.framework_attr[NUM_HEADS]

        if KEY_DIM in mha_node.framework_attr:
            self.kdim = mha_node.framework_attr[KEY_DIM]
        else:
            self.kdim = False

        if VALUE_DIM in mha_node.framework_attr:
            self.vdim = mha_node.framework_attr[VALUE_DIM]
        else:
            self.vdim = False

        self.qdim = int(self.embed_dim / self.num_heads)

        self.q_input, self.k_input, self.v_input = mha_node.input_shape

        # check for input correctness
        assert self.q_input[0] == self.k_input[0] == self.v_input[0], "Batch size must be equal to all inputs"
        assert self.k_input[1] == self.v_input[1], "key's sequence dim do not match value's"
        assert self.kdim == self.k_input[2], "key's embed dim do not match excepted"
        assert self.vdim == self.v_input[2], "value's embed dim do not match excepted"

        self.kv_seq = self.k_input[1]
        self.q_seq = self.q_input[1]

        self.q_input_shape = tuple(self.q_input)
        self.q_transpose_in_shape = tuple([self.q_input[0], self.embed_dim, self.q_seq])
        self.q_reshape_in_shape = tuple([self.q_input[0], self.num_heads, self.qdim, self.q_seq])
        self.q_transpose_shape = tuple([self.q_input[0], self.num_heads, self.q_seq, self.qdim])
        self.q_split_shape = tuple([self.q_input[0], 1, self.q_seq, self.qdim])
        self.q_head_shape = tuple([self.q_input[0], self.q_seq, self.qdim])

        self.k_input_shape = tuple(self.k_input)
        self.k_transpose_in_shape = tuple([self.k_input[0], self.kdim, self.kv_seq])
        self.k_proj_in_shape = tuple([self.k_input[0], self.embed_dim, self.kv_seq])
        self.k_reshape_in_shape = tuple([self.k_input[0], self.num_heads, self.qdim, self.kv_seq])
        self.k_split_shape = tuple([self.k_input[0], 1, self.qdim, self.kv_seq])
        self.k_head_shape = tuple([self.k_input[0], self.qdim, self.kv_seq])

        self.v_input_shape = tuple(self.v_input)
        self.v_transpose_in_shape = tuple([self.v_input[0], self.vdim, self.kv_seq])
        self.v_proj_in_shape = tuple([self.v_input[0], self.embed_dim, self.kv_seq])
        self.v_reshape_in_shape = tuple([self.v_input[0], self.num_heads, self.qdim, self.kv_seq])
        self.v_transpose_shape = tuple([self.v_input[0], self.num_heads, self.kv_seq, self.qdim])
        self.v_split_shape = tuple([self.v_input[0], 1, self.kv_seq, self.qdim])
        self.v_head_shape = tuple([self.v_input[0], self.kv_seq, self.qdim])

        self.attn_mat_shape = tuple([self.q_input[0], 1, self.q_seq, self.kv_seq])
        self.attn_shape = self.q_split_shape

        self.attn_cat_shape = tuple([self.q_input[0], self.num_heads, self.q_seq, self.qdim])
        self.attn_transpose_shape = tuple([self.q_input[0], self.q_seq, self.num_heads, self.qdim])
        self.attn_reshape_shape = self.q_input_shape

        self.transpose_out_proj_shape = self.q_transpose_in_shape
        self.output_shape = self.q_input_shape


class MultiHeadAttentionDecomposition(common.BaseSubstitution):
    """
    Removes a MultiHeadAttention node from the graph,
    and replaces it with a compatible graph that consists of Conv, MatMul, Softmax and stacked layers.
    """

    def __init__(self):
        """
        Matches MultiHeadAttention node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(nn.MultiheadAttention))

    def _project_input(self,
                       graph: Graph,
                       mha_node: BaseNode,
                       params: MHAParams) -> List[BaseNode]:
        """
        This method creates the nodes required to project q, k, v
        We implement the projection as Conv1d
        Due to the above we add transpose node to each of the inputs in order to swap the channel axis according
        to Conv1d expected input shape
        We describe below the shape transformation of each input (q, k, v) from the input shape,
        to transposing and projecting

        Args:
            graph: Graph to apply the substitution on.
            mha_node: MHA node.
            params: MHAnode params.

        Returns:
            List of nodes transposing the inputs and nodes after input projection.
        """
        # q shape transformation:
        # (B, q_seq, q_dim*n_h) --> (B, q_dim*n_h, q_seq) --> (B, q_dim*n_h, q_seq)
        #
        # k shape transformation:
        # (B, kv_seq, k_dim) --> (B, k_dim, kv_seq) --> (B, q_dim*n_h, kv_seq)
        #
        # v shape transformation:
        # (B, kv_seq, v_dim) --> (B, v_dim, kv_seq) --> (B, q_dim*n_h, kv_seq)

        # The scaling const in MHA, we implement the scaling in q input projection weight
        factor = (params.qdim ** -0.5)

        # In some cases (where k_dim and v_dim params isn't supplied) the input projection weight
        # located under 'in_proj_weight' to all inputs (q, k, v) and needs to be separated by 3 inputs
        if IN_PROJ_WEIGHT in mha_node.weights.keys():
            in_proj_w = np.expand_dims(mha_node.get_weights_by_keys(IN_PROJ_WEIGHT).copy(), -1)
            qk, kk, vk = np.split(in_proj_w, 3)
            qk = qk * factor
        # In other cases the weight of the input projection located under 'q_proj_weight' for q input projection and
        # 'k_proj_weight' for k input projection etc
        else:
            qk = np.expand_dims(mha_node.get_weights_by_keys(Q_PROJ_WEIGHT).copy() * factor, -1)
            kk = np.expand_dims(mha_node.get_weights_by_keys(K_PROJ_WEIGHT).copy(), -1)
            vk = np.expand_dims(mha_node.get_weights_by_keys(V_PROJ_WEIGHT).copy(), -1)
        # If the input projection has bias, it'll be located under  'in_proj_bias' for all inputs (q, k, v)
        # and needs to be separated by 3 inputs
        if IN_PROJ_BIAS in mha_node.weights.keys():
            in_proj_bias = mha_node.get_weights_by_keys(IN_PROJ_BIAS).copy()
            qb, kb, vb = np.split(in_proj_bias, 3)
            qb = qb * factor
            in_bias = True
            q_node_weights = {KERNEL: qk, BIAS: qb}
            k_node_weights = {KERNEL: kk, BIAS: kb}
            v_node_weights = {KERNEL: vk, BIAS: vb}
        else:
            in_bias = False
            q_node_weights = {KERNEL: qk}
            k_node_weights = {KERNEL: kk}
            v_node_weights = {KERNEL: vk}

        # project query, key, value

        query_name = f'{mha_node.name}_query_in_proj'
        # Due to the input projection implementation as Conv1d we need to swap the channel axis according from
        # last axis to the second
        # Transpose q second and last axis
        # (B, q_seq, q_dim*n_h) --> (B, q_dim*n_h, q_seq)
        q_transpose_node = FunctionalNode(name=query_name + '_transpose',
                                          framework_attr={},
                                          input_shape=params.q_input_shape,
                                          output_shape=params.q_transpose_in_shape,
                                          weights={},
                                          layer_class=torch.transpose,
                                          op_call_args=[1, 2],
                                          op_call_kwargs={},
                                          functional_op=torch.transpose)
        graph.add_node(q_transpose_node)

        # q doesn't change his shape during projection
        # (B, q_dim*n_h, q_seq)  --> (B, q_dim*n_h, q_seq)
        q_node = BaseNode(name=query_name,
                          framework_attr={IN_CHANNELS: params.embed_dim, OUT_CHANNELS: params.embed_dim,
                                          KERNEL_SIZE: 1, BIAS: in_bias},
                          input_shape=params.q_transpose_in_shape,
                          output_shape=params.q_transpose_in_shape,
                          weights=q_node_weights,
                          layer_class=nn.Conv1d)
        graph.add_node_with_in_edges(q_node, [q_transpose_node])

        key_name = f'{mha_node.name}_key_in_proj'
        # Due to the input projection implementation as Conv1d we need to swap the channel axis according from
        # last axis to the second
        # Transpose k second and last axis
        # (B, k_seq, k_dim) --> (B, k_dim, k_seq)
        k_transpose_node = FunctionalNode(name=key_name + '_transpose',
                                          framework_attr={},
                                          input_shape=params.k_input_shape,
                                          output_shape=params.k_transpose_in_shape,
                                          weights={},
                                          layer_class=torch.transpose,
                                          op_call_args=[1, 2],
                                          op_call_kwargs={},
                                          functional_op=torch.transpose)
        graph.add_node(k_transpose_node)

        # k shape is changing due to projection
        # (B, k_dim, k_seq) --> (B, q_dim*n_h, k_seq)
        k_node = BaseNode(name=key_name,
                          framework_attr={IN_CHANNELS: params.kdim, OUT_CHANNELS: params.embed_dim,
                                          KERNEL_SIZE: 1, BIAS: in_bias},
                          input_shape=params.k_transpose_in_shape,
                          output_shape=params.k_proj_in_shape,
                          weights=k_node_weights,
                          layer_class=nn.Conv1d)
        graph.add_node_with_in_edges(k_node, [k_transpose_node])

        value_name = f'{mha_node.name}_value_in_proj'
        # Due to the input projection implementation as Conv1d we need to swap the channel axis according from
        # last axis to the second
        # Transpose v second and last axis
        # (B, v_seq, v_dim) --> (B, v_dim, v_seq)
        v_transpose_node = FunctionalNode(name=value_name + '_transpose',
                                          framework_attr={},
                                          input_shape=params.v_input_shape,
                                          output_shape=params.v_transpose_in_shape,
                                          weights={},
                                          layer_class=torch.transpose,
                                          op_call_args=[1, 2],
                                          op_call_kwargs={},
                                          functional_op=torch.transpose)
        graph.add_node(v_transpose_node)

        # v shape is changing due to projection
        # (B, v_seq, v_dim) --> (B, q_dim*n_h, v_seq)
        v_node = BaseNode(name=value_name,
                          framework_attr={IN_CHANNELS: params.vdim, OUT_CHANNELS: params.embed_dim,
                                          KERNEL_SIZE: 1, BIAS: in_bias},
                          input_shape=params.v_transpose_in_shape,
                          output_shape=params.v_proj_in_shape,
                          weights=v_node_weights,
                          layer_class=nn.Conv1d)
        graph.add_node_with_in_edges(v_node, [v_transpose_node])

        return q_transpose_node, k_transpose_node, v_transpose_node, q_node, k_node, v_node

    @staticmethod
    def _arrange_before_split(graph: Graph,
                              mha_node: BaseNode,
                              q_node: BaseNode,
                              k_node: BaseNode,
                              v_node: BaseNode,
                              params: MHAParams) -> List[BaseNode]:
        """
        This method creates the nodes required for arranging the shapes of q, k, v, after
        the input projection, before the split by head operation.

        Args:
            graph: Graph to apply the substitution on.
            mha_node: MHA node.
            q_node: query node after input projection.
            k_node: key node after input projection.
            v_node: value node after input projection.
            params: MHAnode params.

        Returns:
            List of nodes after shape arranging.
        """
        # (B, q_dim*n_h, q_seq)  --> (B, n_h, q_dim, q_seq)  --> (B, n_h, q_seq, q_dim)
        # (B, q_dim*n_h, kv_seq) -->  (B, n_h, q_dim, kv_seq)
        # (B, q_dim*n_h, kv_seq) -->  (B, n_h, q_dim, kv_seq) --> (B, n_h, kv_seq, q_dim)

        query_name = f'{mha_node.name}_query'
        # (B, q_dim*n_h, q_seq) --> (B, n_h, q_dim, q_seq)
        q_reshape_node = FunctionalNode(name=query_name + '_reshape',
                                        framework_attr={},
                                        input_shape=params.q_transpose_in_shape,
                                        output_shape=params.q_reshape_in_shape,
                                        weights={},
                                        layer_class=torch.reshape,
                                        op_call_args=[params.q_reshape_in_shape],
                                        op_call_kwargs={},
                                        functional_op=torch.reshape)
        graph.add_node_with_in_edges(q_reshape_node, [q_node])

        # (B, n_h, q_dim, q_seq) --> (B, n_h, q_seq, q_dim)
        q_transpose_node = FunctionalNode(name=query_name + '_transpose',
                                          framework_attr={},
                                          input_shape=params.q_reshape_in_shape,
                                          output_shape=params.q_transpose_shape,
                                          weights={},
                                          layer_class=torch.transpose,
                                          op_call_args=[2, 3],
                                          op_call_kwargs={},
                                          functional_op=torch.transpose)
        graph.add_node_with_in_edges(q_transpose_node, [q_reshape_node])

        key_name = f'{mha_node.name}_key'
        # (B, q_dim*n_h, kv_seq) -->  (B, n_h, q_dim, kv_seq)
        k_reshape_node = FunctionalNode(name=key_name + '_reshape',
                                        framework_attr={},
                                        input_shape=params.k_proj_in_shape,
                                        output_shape=params.k_reshape_in_shape,
                                        weights={},
                                        layer_class=torch.reshape,
                                        op_call_args=[params.k_reshape_in_shape],
                                        op_call_kwargs={},
                                        functional_op=torch.reshape)
        graph.add_node_with_in_edges(k_reshape_node, [k_node])

        value_name = f'{mha_node.name}_value'
        # (B, q_dim*n_h, kv_seq) -->  (B, n_h, q_dim, kv_seq)
        v_reshape_node = FunctionalNode(name=value_name + '_reshape',
                                        framework_attr={},
                                        input_shape=params.v_proj_in_shape,
                                        output_shape=params.v_reshape_in_shape,
                                        weights={},
                                        layer_class=torch.reshape,
                                        op_call_args=[params.v_reshape_in_shape],
                                        op_call_kwargs={},
                                        functional_op=torch.reshape)
        graph.add_node_with_in_edges(v_reshape_node, [v_node])

        # (B, n_h, q_dim, kv_seq) --> (B, n_h, kv_seq, q_dim)
        v_transpose_node = FunctionalNode(name=value_name + '_transpose',
                                          framework_attr={},
                                          input_shape=params.v_reshape_in_shape,
                                          output_shape=params.v_transpose_shape,
                                          weights={},
                                          layer_class=torch.transpose,
                                          op_call_args=[2, 3],
                                          op_call_kwargs={},
                                          functional_op=torch.transpose)
        graph.add_node_with_in_edges(v_transpose_node, [v_reshape_node])

        return q_transpose_node, k_reshape_node, v_transpose_node

    @staticmethod
    def _split_projected(graph: Graph,
                         name: str,
                         q_transpose_node: BaseNode,
                         k_reshape_node: BaseNode,
                         v_transpose_node: BaseNode,
                         params: MHAParams) -> List[BaseNode]:
        """
        This method creates the nodes required for splitting q, k, v to query, key and value per head
        (total of num_heads q, k and v).

        Args:
            graph: Graph to apply the substitution on.
            name: MHA node name.
            q_transpose_node: query node after shape arranging.
            k_reshape_node: key node after shape arranging.
            v_transpose_node: value node after shape arranging.
            params: MHAnode params.

        Returns:
            List of nodes after splitting.
        """
        # Split q, k, v to query, key and value per head (total of num_heads q, k and v)
        # (B, n_h, q_seq, q_dim) --> (B, 1, q_seq, q_dim)* n_h
        # (B, n_h, q_dim, kv_seq) --> (B, 1, q_dim, kv_seq) * n_h
        # (B, n_h, kv_seq, q_dim) --> (B, 1, kv_seq, q_dim) * n_h

        query_name = f'{name}_query'
        # (B, n_h, q_seq, q_dim) --> (B, 1, q_seq, q_dim) * n_h
        q_split_node = FunctionalNode(name=query_name + '_split',
                                      framework_attr={DIM: 1},
                                      input_shape=params.q_transpose_shape,
                                      output_shape=[params.q_split_shape] * params.num_heads,
                                      weights={},
                                      layer_class=torch.split,
                                      op_call_args=[1],
                                      op_call_kwargs={DIM: 1},
                                      functional_op=torch.split)
        graph.add_node_with_in_edges(q_split_node, [q_transpose_node])

        key_name = f'{name}_key'
        # (B, n_h, q_dim, kv_seq) --> (B, 1, q_dim, kv_seq) * n_h
        k_split_node = FunctionalNode(name=key_name + '_split',
                                      framework_attr={DIM: 1},
                                      input_shape=params.k_reshape_in_shape,
                                      output_shape=[params.k_split_shape] * params.num_heads,
                                      weights={},
                                      layer_class=torch.split,
                                      op_call_args=[1],
                                      op_call_kwargs={DIM: 1},
                                      functional_op=torch.split)
        graph.add_node_with_in_edges(k_split_node, [k_reshape_node])

        value_name = f'{name}_value'
        # (B, n_h, kv_seq, q_dim) --> (B, 1, kv_seq, q_dim) * n_h
        v_split_node = FunctionalNode(name=value_name + '_split',
                                      framework_attr={DIM: 1},
                                      input_shape=params.v_reshape_in_shape,
                                      output_shape=[params.v_split_shape] * params.num_heads,
                                      weights={},
                                      layer_class=torch.split,
                                      op_call_args=[1],
                                      op_call_kwargs={DIM: 1},
                                      functional_op=torch.split)
        graph.add_node_with_in_edges(v_split_node, [v_transpose_node])

        return q_split_node, k_split_node, v_split_node

    @staticmethod
    def _calc_attention_head(graph: Graph,
                             q_in_node: BaseNode,
                             k_in_node: BaseNode,
                             v_in_node: BaseNode,
                             mha_node: BaseNode,
                             head_index: int,
                             params: MHAParams) -> BaseNode:
        """
        This method creates the nodes required for attention calc by head

        Args:
            graph: Graph to apply the substitution on.
            q_in_node: query node after shape arranging.
            k_in_node: key node after shape arranging.
            v_in_node: value node after shape arranging.
            mha_node: MHA node.
            head_index: index of the head.
            params: MHAnode params.

        Returns:
            Node after attention calc.
        """
        # Q X K = attn
        # (B, 1, q_seq, q_dim) X (B, 1, q_dim, kv_seq) = (B, 1, q_seq, kv_seq)

        # attn X V = attn_out
        # (B, 1, q_seq, kv_seq) X (B, 1, kv_seq, q_dim) = (B, 1, q_seq, q_dim)

        # (B, 1, q_seq, q_dim) * n_h --> (B, 1, q_seq, kv_seq)
        get_q_name = f'{mha_node.name}_get_q_{head_index}'
        get_q_node = FunctionalNode(name=get_q_name,
                                    framework_attr={},
                                    input_shape=(params.q_split_shape,) * params.num_heads,
                                    output_shape=params.q_split_shape,
                                    weights={},
                                    layer_class=operator.getitem,
                                    op_call_args=[head_index],
                                    op_call_kwargs={},
                                    functional_op=operator.getitem)
        graph.add_node_with_in_edges(get_q_node, [q_in_node], [head_index])

        # (B, 1, q_seq, q_dim) * n_h --> (B, 1, q_seq, kv_seq)
        get_k_name = f'{mha_node.name}_get_k_{head_index}'
        get_k_node = FunctionalNode(name=get_k_name,
                                    framework_attr={},
                                    input_shape=(params.k_split_shape,) * params.num_heads,
                                    output_shape=params.k_split_shape,
                                    weights={},
                                    layer_class=operator.getitem,
                                    op_call_args=[head_index],
                                    op_call_kwargs={},
                                    functional_op=operator.getitem)
        graph.add_node_with_in_edges(get_k_node, [k_in_node], [head_index])

        # (B, 1, q_seq, q_dim) X (B, 1, q_dim, kv_seq) = (B, 1, q_seq, kv_seq)
        matmul_name = f'{mha_node.name}_matmul_{head_index}'
        matmul_node = FunctionalNode(name=matmul_name,
                                     framework_attr={},
                                     input_shape=(params.q_split_shape, params.k_split_shape),
                                     output_shape=params.attn_mat_shape,
                                     weights={},
                                     layer_class=torch.matmul,
                                     op_call_args=[],
                                     op_call_kwargs={},
                                     functional_op=torch.matmul)
        graph.add_node_with_in_edges(matmul_node, [get_q_node, get_k_node])

        # apply softmax on attention matrix
        softmax_name = f'{mha_node.name}_softmax_{head_index}'
        softmax_node = BaseNode(name=softmax_name,
                                framework_attr={DIM: -1},
                                input_shape=params.attn_mat_shape,
                                output_shape=params.attn_mat_shape,
                                weights={},
                                layer_class=nn.Softmax)
        graph.add_node_with_in_edges(softmax_node, [matmul_node])

        # (B, 1, q_seq, q_dim) * n_h --> (B, 1, q_seq, kv_seq)
        get_v_name = f'{mha_node.name}_get_v_{head_index}'
        get_v_node = FunctionalNode(name=get_v_name,
                                    framework_attr={},
                                    input_shape=(params.v_split_shape,) * params.num_heads,
                                    output_shape=params.v_split_shape,
                                    weights={},
                                    layer_class=operator.getitem,
                                    op_call_args=[head_index],
                                    op_call_kwargs={},
                                    functional_op=operator.getitem)
        graph.add_node_with_in_edges(get_v_node, [v_in_node], [head_index])

        # (B, 1, q_seq, kv_seq) X (B, 1, kv_seq, q_dim) = (B, 1, q_seq, q_dim)
        matmulv_name = f'{mha_node.name}_dotv_{head_index}'
        matmulv_node = FunctionalNode(name=matmulv_name,
                                      framework_attr={},
                                      input_shape=(params.attn_mat_shape, params.v_split_shape),
                                      output_shape=params.attn_shape,
                                      weights={},
                                      layer_class=torch.matmul,
                                      op_call_args=[],
                                      op_call_kwargs={},
                                      functional_op=torch.matmul)
        graph.add_node_with_in_edges(matmulv_node, [softmax_node, get_v_node])

        return matmulv_node

    @staticmethod
    def _cat_heads_reshape(graph: Graph,
                           name: str,
                           att_head_output_nodes: List[BaseNode],
                           params: MHAParams) -> BaseNode:
        """
        This method creates the nodes required for concatenating all heads after attention

        Args:
            graph: Graph to apply the substitution on.
            name: MHA node name.
            att_head_output_nodes: list of nodes after attention.
            params: MHAnode params.

        Returns:
            Node after cat and reshape.
        """
        # [(B, 1, q_seq, q_dim)]  * n_h --> (B, n_h, q_seq, q_dim)
        # -->  (B, q_seq, n_h, q_dim)  -->  (B, q_seq, q_dim*n_h)

        # [(B, 1, q_seq, q_dim)]  * n_h --> (B, n_h, q_seq, q_dim)
        cat_node = FunctionalNode(name=f'{name}_cat',
                                  framework_attr={DIM: 1},
                                  input_shape=(params.attn_shape,) * params.num_heads,
                                  output_shape=params.attn_cat_shape,
                                  weights={},
                                  layer_class=torch.cat,
                                  op_call_args=[],
                                  op_call_kwargs={DIM: 1},
                                  functional_op=torch.cat, inputs_as_list=True)
        graph.add_node_with_in_edges(cat_node, att_head_output_nodes)

        # (B, n_h, q_seq, q_dim) -->  (B, q_seq, n_h, q_dim)
        transpose_node = FunctionalNode(name=f'{name}_transpose',
                                        framework_attr={},
                                        input_shape=params.attn_cat_shape,
                                        output_shape=params.attn_transpose_shape,
                                        weights={},
                                        layer_class=torch.transpose,
                                        op_call_args=[1, 2],
                                        op_call_kwargs={},
                                        functional_op=torch.transpose)
        graph.add_node_with_in_edges(transpose_node, [cat_node])

        # (B, q_seq, n_h, q_dim)  -->  (B, q_seq, q_dim*n_h)
        attn_reshape_node = FunctionalNode(name=f'{name}_attn_reshape',
                                           framework_attr={},
                                           input_shape=params.attn_transpose_shape,
                                           output_shape=params.attn_reshape_shape,
                                           weights={},
                                           layer_class=torch.reshape,
                                           op_call_args=[params.attn_reshape_shape],
                                           op_call_kwargs={},
                                           functional_op=torch.reshape)
        graph.add_node_with_in_edges(attn_reshape_node, [transpose_node])

        return attn_reshape_node

    def _project_output(self,
                        graph: Graph,
                        mha_node: BaseNode,
                        attn_reshape_node: BaseNode,
                        params: MHAParams) -> BaseNode:
        """
        This method creates the nodes required for output projecting

        Args:
            graph: Graph to apply the substitution on.
            mha_node: MHA node.
            attn_reshape_node: attention node.
            params: MHAnode params.

        Returns:
            Node after projection.
        """

        # (B, q_seq, q_dim*n_h) -->  (B, q_dim*n_h, q_seq) -->  (B, q_dim*n_h, q_seq) --> (B, q_seq, q_dim*n_h)

        outk = np.expand_dims(mha_node.get_weights_by_keys(OUT_PROJ_WEIGHT).copy(), -1)
        if OUT_PROJ_BIAS in mha_node.weights.keys():
            outb = mha_node.get_weights_by_keys(OUT_PROJ_BIAS).copy()
            out_bias = True
            out_proj_node_weights = {KERNEL: outk, BIAS: outb}
        else:
            out_bias = False
            out_proj_node_weights = {KERNEL: outk}

        # project out
        out_name = f'{mha_node.name}_project_out'

        # transpose proj out
        # (B, q_seq, q_dim*n_h) -->  (B, q_dim*n_h, q_seq)
        transpose_node = FunctionalNode(name=out_name + '_transpose',
                                        framework_attr={},
                                        input_shape=params.attn_reshape_shape,
                                        output_shape=params.transpose_out_proj_shape,
                                        weights={},
                                        layer_class=torch.transpose,
                                        op_call_args=[1, 2],
                                        op_call_kwargs={},
                                        functional_op=torch.transpose)
        graph.add_node_with_in_edges(transpose_node, [attn_reshape_node])

        # (B, q_dim*n_h, q_seq) -->  (B, q_dim*n_h, q_seq)
        proj_out_node = BaseNode(name=out_name,
                                 framework_attr={IN_CHANNELS: params.embed_dim, OUT_CHANNELS: params.embed_dim,
                                                 KERNEL_SIZE: 1, BIAS: out_bias},
                                 input_shape=params.transpose_out_proj_shape,
                                 output_shape=params.transpose_out_proj_shape,
                                 weights=out_proj_node_weights,
                                 layer_class=nn.Conv1d)
        graph.add_node_with_in_edges(proj_out_node, [transpose_node])

        # transpose output
        # (B, q_dim*n_h, q_seq) -->  (B, q_seq, q_dim*n_h)
        transpose_out_node = FunctionalNode(name=f'{mha_node.name}_transpose_output',
                                            framework_attr={},
                                            input_shape=params.transpose_out_proj_shape,
                                            output_shape=params.output_shape,
                                            weights={},
                                            layer_class=torch.transpose,
                                            op_call_args=[1, 2],
                                            op_call_kwargs={},
                                            functional_op=torch.transpose)
        graph.add_node_with_in_edges(transpose_out_node, [proj_out_node])
        return transpose_out_node

    @staticmethod
    def _connect_to_graph(graph: Graph,
                          mha_node: BaseNode,
                          q_node: BaseNode,
                          k_node: BaseNode,
                          v_node: BaseNode,
                          output_permute_node: BaseNode):
        """
        connect subgraph to input graph
        Args:
            graph: input graph
            mha_node: MHA node to substitute inputs and outputs with
            q_node: 1st input to MHA node
            k_node: 2nd input to MHA node
            v_node: 3rd input to MHA node
            output_permute_node: output node of MHA node
        """
        query_in_edge, key_in_edge, value_in_edge = graph.in_edges(mha_node)
        graph.add_edge(query_in_edge[0], q_node, **graph.get_edge_data(*query_in_edge, 0))
        graph.add_edge(key_in_edge[0], k_node, **graph.get_edge_data(*key_in_edge, 0))
        graph.add_edge(value_in_edge[0], v_node, **graph.get_edge_data(*value_in_edge, 0))
        graph.remove_edge(query_in_edge[0], mha_node)
        graph.remove_edge(key_in_edge[0], mha_node)
        graph.remove_edge(value_in_edge[0], mha_node)
        graph.reconnect_out_edges(current_node=mha_node, new_node=output_permute_node)

    def substitute(self,
                   graph: Graph,
                   mha_node: BaseNode) -> Graph:
        """
        connect subgraph to input graph
        Args:
            graph: input graph
            mha_node: MHA node to substitute inputs and outputs with
        Returns:
            Graph after applying the substitution.
        """

        if mha_node.reuse:
            raise Exception("MCT doesn't support reuse of MultiHeadAttention layer")
        params = MHAParams(mha_node)

        # project
        # (B, q_seq, q_dim*n_h) --> (B, q_dim*n_h, q_seq)
        # (B, kv_seq, k_dim) --> (B, q_dim*n_h, kv_seq)
        # (B, kv_seq, v_dim) --> (B, q_dim*n_h, kv_seq)
        q_transpose_node, k_transpose_node, v_transpose_node, \
        q_node, k_node, v_node = self._project_input(graph, mha_node, params)

        # Arrange k, q, v before slicing per head
        # (B, q_dim*n_h, q_seq)  --> (B, n_h, q_seq, q_dim)
        # (B, q_dim*n_h, kv_seq) --> (B, n_h, q_dim, kv_seq)
        # (B, q_dim*n_h, kv_seq) --> (B, n_h, kv_seq, q_dim)
        q_fixed_node, k_fixed_node, v_fixed_node = self._arrange_before_split(graph, mha_node, q_node, k_node, v_node,
                                                                              params)
        # (B, n_h, q_seq, q_dim) --> (B, 1, q_seq, q_dim) * 3
        # (B, n_h, q_dim, kv_seq) --> (B, 1, q_dim, kv_seq) * 3
        # (B, n_h, kv_seq, q_dim) --> (B, 1, kv_seq, q_dim) * 3
        q_split_node, k_split_node, v_split_node = self._split_projected(graph, mha_node.name, q_fixed_node,
                                                                         k_fixed_node, v_fixed_node, params)

        att_head_output_nodes = []
        for h in range(params.num_heads):
            # (B, 1, q_seq, q_dim) X (B, 1, q_dim, kv_seq) = (B, 1, q_seq, kv_seq)
            # Apply Softmax
            # (B, 1, q_seq, kv_seq) X (B, 1, kv_seq, q_dim) = (B, 1, q_seq, q_dim)
            dotv_node = self._calc_attention_head(graph, q_split_node, k_split_node, v_split_node,
                                                  mha_node, h, params)
            # [(B, 1, q_seq, q_dim)] * n_h
            att_head_output_nodes.append(dotv_node)

        # [(B, 1, q_seq, q_dim)]  * n_h -->  (B, q_seq, q_dim*n_h)
        attn_reshape_node = self._cat_heads_reshape(graph, mha_node.name, att_head_output_nodes, params)

        # (B, q_seq, q_dim*n_h) --> (B, q_seq, q_dim*n_h)
        proj_output = self._project_output(graph, mha_node, attn_reshape_node, params)

        # connect edges to new nodes
        self._connect_to_graph(graph, mha_node, q_transpose_node, k_transpose_node, v_transpose_node, proj_output)

        # Finally, remove the MHA node
        graph.remove_node(mha_node, new_graph_outputs=[OutTensor(proj_output, 0)])

        return graph
