# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MultiHeadAttention, Dense, Softmax, Concatenate, Dot, Permute, Multiply

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.keras.constants import KERNEL, DEPTHWISE_KERNEL, BIAS, KERNEL_SIZE, PADDING, \
    STRIDES, USE_BIAS, LINEAR, ACTIVATION, TRAINABLE, FILTERS, PAD_VALID

POINTWISE_KERNEL = 'pointwise_kernel'
DEPTH_MULTIPLIER = 'depth_multiplier'
DATA_FORMAT = 'data_format'
DILATION_RATE = 'dilation_rate'
DEPTHWISE_INITIALIZER = 'depthwise_initializer'
DEPTHWISE_REGULARIZER = 'depthwise_regularizer'
DEPTHWISE_CONSTRAINT = 'depthwise_constraint'
BIAS_INITIALIZER = 'bias_initializer'
BIAS_REGULARIZER = 'bias_regularizer'
BIAS_CONSTRAINT = 'bias_constraint'
ACTIVITY_REGULARIZER = 'activity_regularizer'
KERNEL_INITIALIZER = 'kernel_initializer'
SEPARABLE_PW_KERNEL_INITIALIZER = 'pointwise_initializer'
KERNEL_REGULARIZER = 'kernel_regularizer'
SEPARABLE_PW_KERNEL_REGULARIZER = 'pointwise_regularizer'
KERNEL_CONSTRAINT = 'kernel_constraint'
SEPARABLE_PW_KERNEL_CONSTRAINT = 'pointwise_constraint'


class MultiHeadAttentionDecomposition(common.BaseSubstitution):
    """
    Removes a MultiHeadAttention node from the graph,
    and replaces it with a compatible graph that consists Dense, Dot, Softmax and Concatenate layers
    (and tf.transpose)
    """

    def __init__(self):
        """
        Matches MultiHeadAttention node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(MultiHeadAttention))

    def substitute(self,
                   graph: Graph,
                   mha_node: BaseNode) -> Graph:
        """
        Remove a SeparableConv2D node from the graph, and replace it with two equivalent nodes: DepthwiseConv2D
        and Conv2D. The SeparableConv2D attributes are split to relevant attributes for each node.

        Args:
            graph: Graph we apply the substitution on.
            separable_node: Separable node to replace with a depthwise and pointwise nodes.

        Returns:
            Graph after applying the substitution.
        """

        # multi_head_attention_1 / query / kernel: 0(8, 3, 4)
        # multi_head_attention_1 / query / bias: 0(3, 4)
        # multi_head_attention_1 / key / kernel: 0(8, 3, 4)
        # multi_head_attention_1 / key / bias: 0(3, 4)
        # multi_head_attention_1 / value / kernel: 0(8, 3, 6)
        # multi_head_attention_1 / value / bias: 0(3, 6)
        # multi_head_attention_1 / attention_output / kernel: 0(3, 6, 8)
        # multi_head_attention_1 / attention_output / bias: 0(8, )

        query_nodes, key_nodes, value_nodes, att_head_output_nodes = [], [], [], []

        # MHA params:
        num_heads = mha_node.framework_attr['num_heads']
        use_bias = mha_node.framework_attr['use_bias']
        key_dim = mha_node.framework_attr['key_dim']
        value_dim = mha_node.framework_attr['value_dim']
        query_shape = mha_node.framework_attr['query_shape']
        key_shape = mha_node.framework_attr['key_shape']
        value_shape = mha_node.framework_attr['value_shape']
        d_model = value_shape[-1]
        sequence_length = value_shape[1]
        query_proj_shape = mha_node.framework_attr['query_shape'][:-1] + [key_dim]
        key_proj_shape = mha_node.framework_attr['key_shape'][:-1] + [key_dim]
        value_proj_shape = mha_node.framework_attr['value_shape'][:-1] + [value_dim]
        concat_shape = mha_node.framework_attr['value_shape'][:-1] + [value_dim*num_heads]

        reorder = lambda x, order: tuple([x[_i] for _i in order])

        # matmul_output = layers.Dot(axes=(1, 2))([layers.Permute([2, 1])(q), k])
        # attention_matrix = layers.Softmax()(layers.Multiply()([matmul_output, norm_factor]))
        # return layers.Permute([2, 1])(layers.Dot(axes=(1, 2))([v, attention_matrix])), attention_matrix

        # Generate nodes for attention heads:
        # TODO: define constant
        norm_factor = mha_node.weights[mha_node.name + '/query/kernel:0'].shape[-1] ** -0.5
        for h in range(num_heads):
            qk = mha_node.weights[mha_node.name + '/query/kernel:0'][:, h, :]
            kk = mha_node.weights[mha_node.name + '/key/kernel:0'][:, h, :]
            vk = mha_node.weights[mha_node.name + '/value/kernel:0'][:, h, :]
            qb = mha_node.weights[mha_node.name + '/query/bias:0'][h, :]
            kb = mha_node.weights[mha_node.name + '/key/bias:0'][h, :]
            vb = mha_node.weights[mha_node.name + '/value/bias:0'][h, :]
            # create new nodes:

            # project quary, key & value inputs to key_dim, key_dim & value_dim respectively
            q_node = BaseNode(f'{mha_node.name}_query_{h}', {'units': key_dim, 'use_bias': use_bias},
                              query_shape, query_proj_shape, {KERNEL: qk, BIAS: qb}, Dense,
                              reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            query_nodes.append(q_node)
            graph.add_node(q_node)
            k_node = BaseNode(f'{mha_node.name}_key_{h}', {'units': key_dim, 'use_bias': use_bias},
                              key_shape, key_proj_shape, {KERNEL: kk, BIAS: kb}, Dense,
                              reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            key_nodes.append(k_node)
            graph.add_node(k_node)
            v_node = BaseNode(f'{mha_node.name}_value_{h}', {'units': value_dim, 'use_bias': use_bias},
                              value_shape, value_proj_shape, {KERNEL: vk, BIAS: vb}, Dense,
                              reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            value_nodes.append(v_node)
            graph.add_node(v_node)

            # calculate attention matrix:
            # apply tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]) as layers.Dot(axes=(1, 2))([layers.Permute([2, 1])(q), k])
            permute_node = BaseNode(f'{mha_node.name}_permute_{h}', {'dims': (2, 1)},
                                    query_shape, reorder(query_shape, (0, 2, 1)), {}, Permute,
                                    reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(permute_node, [q_node])
            # graph.add_node(permute_node)
            # graph.add_edge(q_node, permute_node, source_index=0, sink_index=0)

            dot_node = BaseNode(f'{mha_node.name}_dot_{h}', {'axes': (1, 2)},
                                key_shape, tuple([key_shape[0]] + [sequence_length, sequence_length]), {}, Dot,
                                reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(dot_node, [permute_node, k_node])
            # graph.add_node(dot_node)
            # graph.add_edge(permute_node, dot_node, source_index=0, sink_index=0)
            # graph.add_edge(k_node, dot_node, source_index=0, sink_index=1)

            # apply softmax on attention matric
            softmax_node = BaseNode(f'{mha_node.name}_softmax_{h}', {},
                                    tuple([key_shape[0]] + [sequence_length, sequence_length]), tuple([key_shape[0]] + [sequence_length, sequence_length]),
                                    {}, Softmax, reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(softmax_node, [dot_node])
            # graph.add_node(softmax_node)
            # graph.add_edge(dot_node, softmax_node, source_index=0, sink_index=0)

            # calculate matmul(attention_matrix, projected_value) TODO: how to handle the input shape of a node with 2 inputs??
            dot_node = BaseNode(f'{mha_node.name}_dotv_{h}', {'axes': (1, 2)},
                                tuple([key_shape[0]] + [sequence_length, sequence_length]), reorder(value_proj_shape, (0, 2, 1)),
                                {}, Dot, reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(dot_node, [v_node, softmax_node])
            # graph.add_node(dot_node)
            # graph.add_edge(v_node, dot_node, source_index=0, sink_index=0)
            # graph.add_edge(softmax_node, dot_node, source_index=0, sink_index=1)

            permute_node = BaseNode(f'{mha_node.name}_permute_out_{h}', {'dims': (2, 1)},
                                    reorder(value_proj_shape, (0, 2, 1)), value_proj_shape, {}, Permute,
                                    reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(permute_node, [dot_node])
            # graph.add_node(permute_node)
            # graph.add_edge(dot_node, permute_node, source_index=0, sink_index=0)

            att_head_output_nodes.append(permute_node)

        # concatenate all attention heads
        concat_node = BaseNode(f'{mha_node.name}_concat', {'dims': (2, 1)},
                               value_proj_shape, concat_shape, {}, Concatenate,
                               reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
        graph.add_node_with_in_edges(concat_node, att_head_output_nodes)
        # graph.add_node(concat_node)
        # for h in range(num_heads):
        #     graph.add_edge(att_head_output_nodes[h], concat_node, source_index=0, sink_index=h)

        w_out = mha_node.weights[mha_node.name + '/attention_output/kernel:0'].reshape((-1, d_model))
        b_out = mha_node.weights[mha_node.name + '/attention_output/bias:0']
        output_dense = BaseNode(f'{mha_node.name}_output_dense', {'units': d_model, 'use_bias': use_bias},
                                query_shape, query_proj_shape, {KERNEL: w_out, BIAS: b_out}, Dense,
                                reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
        graph.add_node_with_in_edges(output_dense, [concat_node])
        # graph.add_node(output_dense)
        # graph.add_edge(concat_node, output_dense, source_index=0, sink_index=0)

        # connect edges to new nodes
        query_in_edge, key_value_in_edge = graph.in_edges(mha_node)
        for h in range(num_heads):
            graph.add_edge(query_in_edge[0], query_nodes[h], **graph.get_edge_data(*query_in_edge, 0))
            graph.add_edge(key_value_in_edge[0], key_nodes[h], **graph.get_edge_data(*key_value_in_edge, 0))
            graph.add_edge(key_value_in_edge[0], value_nodes[h], **graph.get_edge_data(*key_value_in_edge, 0))
        graph.remove_edge(query_in_edge[0], mha_node)
        graph.remove_edge(key_value_in_edge[0], mha_node)
        graph.reconnect_out_edges(current_node=mha_node, new_node=output_dense)

        graph.remove_node(mha_node, new_graph_outputs=[OutTensor(output_dense, 0)])

        return graph
