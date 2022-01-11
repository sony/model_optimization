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
if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda
from tensorflow.keras.layers import MultiHeadAttention, Dense, Softmax, Concatenate, Dot, Reshape, Permute

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.keras.constants import KERNEL, BIAS


class MultiHeadAttentionDecomposition(common.BaseSubstitution):
    """
    Removes a MultiHeadAttention node from the graph,
    and replaces it with a compatible graph that consists of Dense,
     Dot, Softmax and Concatenate layers
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
        Removes a MultiHeadAttention node from the graph, and replaces it with
         a compatible graph that consists of Dense, Dot, Softmax and Concatenate layers

        Args:
            graph: Graph we apply the substitution on.
            mha_node: MultiHeadAttention node to replace.

        Returns:
            Graph after applying the substitution.
        """

        query_nodes, key_nodes, value_nodes, att_head_output_nodes = [], [], [], []

        # MHA params:
        num_heads = mha_node.framework_attr['num_heads']
        use_bias = mha_node.framework_attr['use_bias']
        query_key_dim = mha_node.framework_attr['key_dim']
        value_dim = mha_node.framework_attr['value_dim']
        query_shape = tuple(mha_node.framework_attr['query_shape'])
        key_shape = tuple(mha_node.framework_attr['key_shape'])
        value_shape = tuple(mha_node.framework_attr['value_shape'])
        output_dim = mha_node.framework_attr['output_shape']
        d_model = query_shape[-1] if output_dim is None else output_dim
        attention_axes = tuple(mha_node.framework_attr['attention_axes'])

        iter_axes, iter_axes_prod, q_att_axes_prod, kv_att_axes_prod = [], 1, 1, 1
        for i, (aq, ak) in enumerate(zip(query_shape[1:-1], key_shape[1:-1])):
            if i+1 in attention_axes:
                q_att_axes_prod = q_att_axes_prod * aq
                kv_att_axes_prod = kv_att_axes_prod * ak
            else:
                iter_axes.append(i+1)
                iter_axes_prod = iter_axes_prod * aq
        perm_dims = [0] + iter_axes + list(attention_axes) + [len(query_shape)-1]
        output_perm_dims = [perm_dims.index(i) for i in range(len(perm_dims))]  # dims to revert perm_dims
        q_perm_shape = tuple([query_shape[i] for i in perm_dims])
        k_perm_shape = tuple([key_shape[i] for i in perm_dims])
        v_perm_shape = tuple([value_shape[i] for i in perm_dims])
        q_reshape_shape = tuple([query_shape[0], q_att_axes_prod, query_shape[-1]])
        k_reshape_shape = tuple([key_shape[0], kv_att_axes_prod, key_shape[-1]])
        v_reshape_shape = tuple([value_shape[0], kv_att_axes_prod, value_shape[-1]])
        query_proj_shape = q_reshape_shape[:-1] + (query_key_dim,)
        key_proj_shape = k_reshape_shape[:-1] + (query_key_dim,)
        value_proj_shape = v_reshape_shape[:-1] + (value_dim,)
        att_matrix_shape = (key_shape[0],) + (q_att_axes_prod, kv_att_axes_prod)

        get_weight_name = lambda w_str: [k for k in mha_node.weights.keys() if w_str in k][0]

        # input permutation and reshape to standard shape: (batch, sequence, channels)
        mha_in_edges = graph.in_edges(mha_node)
        if len(mha_in_edges) == 3:
            # MHA node is called with 3 inputs: Query, Value & Key
            q_permute_node = BaseNode(f'{mha_node.name}_query_input_permute', {'dims': perm_dims[1:]},
                                      query_shape, q_perm_shape, {}, Permute,
                                      reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node(q_permute_node)
            k_permute_node = BaseNode(f'{mha_node.name}_key_input_permute', {'dims': perm_dims[1:]},
                                      key_shape, k_perm_shape, {}, Permute,
                                      reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node(k_permute_node)
            v_permute_node = BaseNode(f'{mha_node.name}_value_input_permute', {'dims': perm_dims[1:]},
                                      value_shape, v_perm_shape, {}, Permute,
                                      reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node(v_permute_node)

            q_reshape_node = FunctionalNode(f'{mha_node.name}_query_input_reshape', {'function': 'reshape'},
                                            q_perm_shape, q_reshape_shape, {}, TFOpLambda,
                                            op_call_args=[(-1,) + q_reshape_shape[1:]], op_call_kwargs={},
                                            reuse=mha_node.reuse, reuse_group=mha_node.reuse_group,
                                            functional_op=tf.reshape)
            graph.add_node_with_in_edges(q_reshape_node, [q_permute_node])
            k_reshape_node = FunctionalNode(f'{mha_node.name}_key_input_reshape', {'function': 'reshape'},
                                            k_perm_shape, k_reshape_shape, {}, TFOpLambda,
                                            op_call_args=[(-1,) + k_reshape_shape[1:]], op_call_kwargs={},
                                            reuse=mha_node.reuse, reuse_group=mha_node.reuse_group,
                                            functional_op=tf.reshape)
            graph.add_node_with_in_edges(k_reshape_node, [k_permute_node])
            v_reshape_node = FunctionalNode(f'{mha_node.name}_value_input_reshape', {'function': 'reshape'},
                                            v_perm_shape, v_reshape_shape, {}, TFOpLambda,
                                            op_call_args=[(-1,) + v_reshape_shape[1:]], op_call_kwargs={},
                                            reuse=mha_node.reuse, reuse_group=mha_node.reuse_group,
                                            functional_op=tf.reshape)
            graph.add_node_with_in_edges(v_reshape_node, [v_permute_node])

        else:
            # MHA node is called with 2 inputs: Query & Value. Key=Value
            q_permute_node = BaseNode(f'{mha_node.name}_query_input_permute', {'dims': perm_dims[1:]},
                                      query_shape, q_perm_shape, {}, Permute,
                                      reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node(q_permute_node)
            v_permute_node = BaseNode(f'{mha_node.name}_value_input_permute', {'dims': perm_dims[1:]},
                                      value_shape, v_perm_shape, {}, Permute,
                                      reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node(v_permute_node)
            q_reshape_node = FunctionalNode(f'{mha_node.name}_query_input_reshape', {'function': 'reshape'},
                                            q_perm_shape, q_reshape_shape, {}, TFOpLambda,
                                            op_call_args=[(-1,) + q_reshape_shape[1:]], op_call_kwargs={},
                                            reuse=mha_node.reuse, reuse_group=mha_node.reuse_group,
                                            functional_op=tf.reshape)
            graph.add_node_with_in_edges(q_reshape_node, [q_permute_node])
            v_reshape_node = FunctionalNode(f'{mha_node.name}_value_input_reshape', {'function': 'reshape'},
                                            v_perm_shape, v_reshape_shape, {}, TFOpLambda,
                                            op_call_args=[(-1,) + v_reshape_shape[1:]], op_call_kwargs={},
                                            reuse=mha_node.reuse, reuse_group=mha_node.reuse_group,
                                            functional_op=tf.reshape)
            graph.add_node_with_in_edges(v_reshape_node, [v_permute_node])
            k_permute_node = v_permute_node
            k_reshape_node = v_reshape_node
            k_reshape_shape = v_reshape_shape

        # Generate nodes for attention heads:
        for h in range(num_heads):
            # add norm factor to query kernel and bias
            qk = mha_node.weights[get_weight_name('/query/kernel')][:, h, :].copy() * (query_key_dim ** -0.5)
            kk = mha_node.weights[get_weight_name('/key/kernel')][:, h, :].copy()
            vk = mha_node.weights[get_weight_name('/value/kernel')][:, h, :].copy()
            qb = mha_node.weights[get_weight_name('/query/bias')][h, :].copy() * (query_key_dim ** -0.5)
            kb = mha_node.weights[get_weight_name('/key/bias')][h, :].copy()
            vb = mha_node.weights[get_weight_name('/value/bias')][h, :].copy()

            # create new nodes:
            # project query, key & value inputs to query_key_dim, query_key_dim & value_dim respectively
            q_node = BaseNode(f'{mha_node.name}_query_{h}', {'units': query_key_dim, 'use_bias': use_bias, 'activation': 'linear'},
                              q_reshape_shape, query_proj_shape, {KERNEL: qk, BIAS: qb}, Dense,
                              reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            # query_nodes.append(q_node)
            # graph.add_node(q_node)
            graph.add_node_with_in_edges(q_node, [q_reshape_node])
            k_node = BaseNode(f'{mha_node.name}_key_{h}', {'units': query_key_dim, 'use_bias': use_bias, 'activation': 'linear'},
                              k_reshape_shape, key_proj_shape, {KERNEL: kk, BIAS: kb}, Dense,
                              reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            # key_nodes.append(k_node)
            # graph.add_node(k_node)
            graph.add_node_with_in_edges(k_node, [k_reshape_node])
            v_node = BaseNode(f'{mha_node.name}_value_{h}', {'units': value_dim, 'use_bias': use_bias, 'activation': 'linear'},
                              v_reshape_shape, value_proj_shape, {KERNEL: vk, BIAS: vb}, Dense,
                              reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            # value_nodes.append(v_node)
            # graph.add_node(v_node)
            graph.add_node_with_in_edges(v_node, [v_reshape_node])

            # calculate attention matrix:
            # apply tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]) as layers.Dot(axes=2)([q, k])
            dot_node = BaseNode(f'{mha_node.name}_dot_{h}', {'axes': 2},
                                (query_proj_shape, key_proj_shape), att_matrix_shape, {}, Dot,
                                reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(dot_node, [q_node, k_node])

            # apply softmax on attention matrix
            softmax_node = BaseNode(f'{mha_node.name}_softmax_{h}', {},
                                    att_matrix_shape, att_matrix_shape,
                                    {}, Softmax, reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(softmax_node, [dot_node])

            dotv_node = BaseNode(f'{mha_node.name}_dotv_{h}', {'axes': (2, 1)},
                                 (att_matrix_shape, value_proj_shape), value_proj_shape,
                                 {}, Dot, reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
            graph.add_node_with_in_edges(dotv_node, [softmax_node, v_node])
            att_head_output_nodes.append(dotv_node)

        # concatenate all attention heads
        concat_shape = (key_shape[0],) + (q_att_axes_prod, value_dim*num_heads)
        output_shape = concat_shape[:-1] + (d_model,)
        concat_node = BaseNode(f'{mha_node.name}_concat', {},
                               (value_proj_shape,)*num_heads, concat_shape, {}, Concatenate,
                               reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
        graph.add_node_with_in_edges(concat_node, att_head_output_nodes)

        w_out = mha_node.weights[get_weight_name('/attention_output/kernel')].copy().reshape((-1, d_model))
        b_out = mha_node.weights[get_weight_name('/attention_output/bias')].copy()
        output_dense = BaseNode(f'{mha_node.name}_output_dense', {'units': d_model, 'use_bias': use_bias, 'activation': 'linear'},
                                concat_shape, output_shape, {KERNEL: w_out, BIAS: b_out}, Dense,
                                reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
        graph.add_node_with_in_edges(output_dense, [concat_node])

        # re-order output to match MHA node output (reshape+permute)
        output_reshape_node = FunctionalNode(f'{mha_node.name}_output_reshape', {'function': 'reshape'},
                                             output_shape, q_perm_shape[:-1] + (d_model,), {}, TFOpLambda,
                                             op_call_args=[(-1,) + q_perm_shape[1:-1] + (d_model,)], op_call_kwargs={},
                                             reuse=mha_node.reuse, reuse_group=mha_node.reuse_group,
                                             functional_op=tf.reshape)
        graph.add_node_with_in_edges(output_reshape_node, [output_dense])
        output_permute_node = BaseNode(f'{mha_node.name}_output_permute', {'dims': output_perm_dims[1:]},
                                       q_perm_shape[:-1] + (d_model,), query_shape[:-1] + (d_model,), {}, Permute,
                                       reuse=mha_node.reuse, reuse_group=mha_node.reuse_group)
        graph.add_node_with_in_edges(output_permute_node, [output_reshape_node])

        # connect edges to new nodes
        if len(mha_in_edges) == 3:
            # MHA node is called with 3 inputs: Query, Value & Key
            query_in_edge, value_in_edge, key_in_edge = graph.in_edges(mha_node)
        else:
            # MHA node is called with 2 inputs: Query & Value. Key=Value
            query_in_edge, value_in_edge = graph.in_edges(mha_node)
            key_in_edge = value_in_edge
        graph.add_edge(query_in_edge[0], q_permute_node, **graph.get_edge_data(*query_in_edge, 0))
        if key_in_edge is not value_in_edge:
            graph.add_edge(key_in_edge[0], k_permute_node, **graph.get_edge_data(*key_in_edge, 0))
        graph.add_edge(value_in_edge[0], v_permute_node, **graph.get_edge_data(*value_in_edge, 0))
        graph.remove_edge(query_in_edge[0], mha_node)
        graph.remove_edge(key_in_edge[0], mha_node)
        if key_in_edge is not value_in_edge:
            graph.remove_edge(value_in_edge[0], mha_node)
        graph.reconnect_out_edges(current_node=mha_node, new_node=output_permute_node)

        graph.remove_node(mha_node, new_graph_outputs=[OutTensor(output_permute_node, 0)])

        return graph
