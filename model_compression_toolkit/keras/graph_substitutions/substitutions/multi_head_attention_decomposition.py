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
from model_compression_toolkit.common.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.keras.constants import KERNEL, BIAS, USE_BIAS, NUM_HEADS, KEY_DIM, VALUE_DIM, \
    QUERY_SHAPE, KEY_SHAPE, VALUE_SHAPE, OUTPUT_SHAPE, ATTENTION_AXES, ACTIVATION, LINEAR, UNITS, AXES, \
    FUNCTION, DIMS, F_RESHAPE, Q_KERNEL, Q_BIAS, K_KERNEL, K_BIAS, V_KERNEL, V_BIAS, OUTPUT_KERNEL, OUTPUT_BIAS


class MHAParams:
    """
    A data class to hold all relevant parameters from the MHA node framework attributes
    """
    def __init__(self, mha_node):
        """
        Extract MHA params from layer attributes
        :param mha_node: MHA node
        """
        self.num_heads = mha_node.framework_attr[NUM_HEADS]
        self.use_bias = mha_node.framework_attr[USE_BIAS]
        self.query_key_dim = mha_node.framework_attr[KEY_DIM]
        self.value_dim = mha_node.framework_attr[VALUE_DIM]
        self.query_shape = tuple(mha_node.framework_attr[QUERY_SHAPE])
        self.key_shape = tuple(mha_node.framework_attr[KEY_SHAPE])
        self.value_shape = tuple(mha_node.framework_attr[VALUE_SHAPE])
        output_dim = mha_node.framework_attr[OUTPUT_SHAPE]
        self.d_model = self.query_shape[-1] if output_dim is None else output_dim
        self.attention_axes = tuple(mha_node.framework_attr[ATTENTION_AXES])

        # compute the parameters for folding the iteration and attention axes
        iter_axes, iter_axes_prod, self.q_att_axes_prod, kv_att_axes_prod = [], 1, 1, 1
        for i, (aq, ak) in enumerate(zip(self.query_shape[1:-1], self.key_shape[1:-1])):
            if i+1 in self.attention_axes:
                self.q_att_axes_prod = self.q_att_axes_prod * aq
                kv_att_axes_prod = kv_att_axes_prod * ak
            else:
                iter_axes.append(i+1)
                iter_axes_prod = iter_axes_prod * aq
        self.perm_dims = [0] + iter_axes + list(self.attention_axes) + [len(self.query_shape)-1]
        self.output_perm_dims = [self.perm_dims.index(i) for i in range(len(self.perm_dims))]  # dims to revert perm_dims
        self.q_perm_shape = tuple([self.query_shape[i] for i in self.perm_dims])
        self.k_perm_shape = tuple([self.key_shape[i] for i in self.perm_dims])
        self.v_perm_shape = tuple([self.value_shape[i] for i in self.perm_dims])
        self.q_reshape_shape = tuple([self.query_shape[0], self.q_att_axes_prod, self.query_shape[-1]])
        self.k_reshape_shape = tuple([self.key_shape[0], kv_att_axes_prod, self.key_shape[-1]])
        self.v_reshape_shape = tuple([self.value_shape[0], kv_att_axes_prod, self.value_shape[-1]])
        self.query_proj_shape = self.q_reshape_shape[:-1] + (self.query_key_dim,)
        self.key_proj_shape = self.k_reshape_shape[:-1] + (self.query_key_dim,)
        self.value_proj_shape = self.v_reshape_shape[:-1] + (self.value_dim,)
        self.att_matrix_shape = (self.key_shape[0],) + (self.q_att_axes_prod, kv_att_axes_prod)
        self.reuse_params = {REUSE: mha_node.reuse, REUSE_GROUP: mha_node.reuse_group}
        self.concat_shape = (self.key_shape[0],) + (self.q_att_axes_prod, self.value_dim*self.num_heads)
        self.output_shape = self.concat_shape[:-1] + (self.d_model,)


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

    @staticmethod
    def _get_weight_by_name(mha_node, w_str):
        return [k for k in mha_node.weights.keys() if w_str in k][0]

    @staticmethod
    def _standarize_input_shapes(graph, name, num_input_edges, params):
        """
        Add Permute and tf.reshape nodes to stadarize the inputs. output shape will be [B, Sequence, Channels],
        where the batch axis contains the iteration axes, if any

        :param graph: input graph
        :param name: MHA node name
        :param num_input_edges: number of input to MHA node. Must be either 2 or 3
        :param params: MHA params object
        :return: Query, Key & Value nodes to use as inputs to subgraph,
                 Standarized Query, Key & Value nodes to connect to the rest of the subgraph
        """
        assert num_input_edges in [2, 3]
        q_permute_node = BaseNode(f'{name}_query_input_permute', {DIMS: params.perm_dims[1:]},
                                  params.query_shape, params.q_perm_shape, {}, Permute, **params.reuse_params)
        graph.add_node(q_permute_node)
        v_permute_node = BaseNode(f'{name}_value_input_permute', {DIMS: params.perm_dims[1:]},
                                  params.value_shape, params.v_perm_shape, {}, Permute, **params.reuse_params)
        graph.add_node(v_permute_node)

        q_reshape_node = FunctionalNode(f'{name}_query_input_reshape', {FUNCTION: F_RESHAPE},
                                        params.q_perm_shape, params.q_reshape_shape, {}, TFOpLambda,
                                        op_call_args=[(-1,) + params.q_reshape_shape[1:]], op_call_kwargs={},
                                        functional_op=tf.reshape, **params.reuse_params)
        graph.add_node_with_in_edges(q_reshape_node, [q_permute_node])
        v_reshape_node = FunctionalNode(f'{name}_value_input_reshape', {FUNCTION: F_RESHAPE},
                                        params.v_perm_shape, params.v_reshape_shape, {}, TFOpLambda,
                                        op_call_args=[(-1,) + params.v_reshape_shape[1:]], op_call_kwargs={},
                                        functional_op=tf.reshape, **params.reuse_params)
        graph.add_node_with_in_edges(v_reshape_node, [v_permute_node])
        if num_input_edges == 3:
            # MHA node is called with 3 inputs: Query, Value & Key
            k_permute_node = BaseNode(f'{name}_key_input_permute', {DIMS: params.perm_dims[1:]},
                                      params.key_shape, params.k_perm_shape, {}, Permute, **params.reuse_params)
            graph.add_node(k_permute_node)
            k_reshape_node = FunctionalNode(f'{name}_key_input_reshape', {FUNCTION: F_RESHAPE},
                                            params.k_perm_shape, params.k_reshape_shape, {}, TFOpLambda,
                                            op_call_args=[(-1,) + params.k_reshape_shape[1:]], op_call_kwargs={},
                                            functional_op=tf.reshape, **params.reuse_params)
            graph.add_node_with_in_edges(k_reshape_node, [k_permute_node])
        else:
            # MHA node is called with 2 inputs: Query & Value. Key=Value
            k_permute_node = v_permute_node
            k_reshape_node = v_reshape_node

        return (q_permute_node, k_permute_node, v_permute_node,
                q_reshape_node, k_reshape_node, v_reshape_node)

    def _calc_attention_head(self, graph, q_reshape_node, k_reshape_node, v_reshape_node, mha_node,
                             head_index, params):
        """
        Generate the attention head subgraph: Dot(softmax(Dot(proj(Q), proj(K))), proj(V))
        :param graph: input graph
        :param q_reshape_node:
        :param k_reshape_node:
        :param v_reshape_node:
        :param mha_node: MHA node
        :param head_index: head index being generated. used to set correct node names
        :param params: MHA params object
        :return: output of attention head
        """
        # add norm factor to query kernel and bias
        factor = (params.query_key_dim ** -0.5)
        qk = mha_node.weights[self._get_weight_by_name(mha_node, Q_KERNEL)][:, head_index, :].copy() * factor
        kk = mha_node.weights[self._get_weight_by_name(mha_node, K_KERNEL)][:, head_index, :].copy()
        vk = mha_node.weights[self._get_weight_by_name(mha_node, V_KERNEL)][:, head_index, :].copy()
        qb = mha_node.weights[self._get_weight_by_name(mha_node, Q_BIAS)][head_index, :].copy() * factor
        kb = mha_node.weights[self._get_weight_by_name(mha_node, K_BIAS)][head_index, :].copy()
        vb = mha_node.weights[self._get_weight_by_name(mha_node, V_BIAS)][head_index, :].copy()

        # create new nodes:
        # project query, key & value inputs to query_key_dim, query_key_dim & value_dim respectively
        q_node = BaseNode(f'{mha_node.name}_query_{head_index}', {UNITS: params.query_key_dim, USE_BIAS: params.use_bias,
                                                                  ACTIVATION: LINEAR},
                          params.q_reshape_shape, params.query_proj_shape, {KERNEL: qk, BIAS: qb}, Dense, **params.reuse_params)
        graph.add_node_with_in_edges(q_node, [q_reshape_node])
        k_node = BaseNode(f'{mha_node.name}_key_{head_index}', {UNITS: params.query_key_dim, USE_BIAS: params.use_bias, ACTIVATION: LINEAR},
                          params.k_reshape_shape, params.key_proj_shape, {KERNEL: kk, BIAS: kb}, Dense, **params.reuse_params)
        graph.add_node_with_in_edges(k_node, [k_reshape_node])
        v_node = BaseNode(f'{mha_node.name}_value_{head_index}', {UNITS: params.value_dim, USE_BIAS: params.use_bias, ACTIVATION: LINEAR},
                          params.v_reshape_shape, params.value_proj_shape, {KERNEL: vk, BIAS: vb}, Dense, **params.reuse_params)
        graph.add_node_with_in_edges(v_node, [v_reshape_node])

        # calculate attention matrix:
        # apply tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]) as layers.Dot(axes=2)([q, k])
        dot_node = BaseNode(f'{mha_node.name}_dot_{head_index}', {AXES: 2},
                            (params.query_proj_shape, params.key_proj_shape), params.att_matrix_shape, {}, Dot,
                            **params.reuse_params)
        graph.add_node_with_in_edges(dot_node, [q_node, k_node])

        # apply softmax on attention matrix
        softmax_node = BaseNode(f'{mha_node.name}_softmax_{head_index}', {},
                                params.att_matrix_shape, params.att_matrix_shape,
                                {}, Softmax, **params.reuse_params)
        graph.add_node_with_in_edges(softmax_node, [dot_node])

        dotv_node = BaseNode(f'{mha_node.name}_dotv_{head_index}', {AXES: (2, 1)},
                             (params.att_matrix_shape, params.value_proj_shape), params.value_proj_shape,
                             {}, Dot, **params.reuse_params)
        graph.add_node_with_in_edges(dotv_node, [softmax_node, v_node])

        return dotv_node

    @staticmethod
    def _concat_heads(graph, name, input_nodes, params):
        """
        concatenate attention heads

        :param graph: input graph
        :param name: MHA node name
        :param input_nodes: all the outputs of the attention heads
        :param params: MHA params object
        :return: concat node
        """
        concat_node = BaseNode(f'{name}_concat', {},
                               (params.value_proj_shape,)*params.num_heads, params.concat_shape, {}, Concatenate,
                               **params.reuse_params)
        graph.add_node_with_in_edges(concat_node, input_nodes)
        return concat_node

    def _project_output(self, graph, mha_node, input_node, params):
        """
        project all attention outputs (after concatenation

        :param graph: input graph
        :param mha_node: MHA node
        :param input_node: concat node
        :param params: MHA params object
        :return: output node of the MHA node
        """
        w_out = mha_node.weights[self._get_weight_by_name(mha_node, OUTPUT_KERNEL)].copy().reshape((-1, params.d_model))
        b_out = mha_node.weights[self._get_weight_by_name(mha_node, OUTPUT_BIAS)].copy()
        output_dense = BaseNode(f'{mha_node.name}_output_dense', {UNITS: params.d_model, USE_BIAS: params.use_bias, ACTIVATION: LINEAR},
                                params.concat_shape, params.output_shape, {KERNEL: w_out, BIAS: b_out}, Dense, **params.reuse_params)
        graph.add_node_with_in_edges(output_dense, [input_node])
        return output_dense

    @staticmethod
    def _destandarize_output_shapes(graph, name, output_node, params):
        """
        return output to original MHA output shape
        :param graph: input graph
        :param name: MHA node name
        :param output_node: output node of output projection
        :param params: MHA params object
        :return: Destandarized output node, which matches in shape the output of the MHA node
        """
        output_reshape_node = FunctionalNode(f'{name}_output_reshape', {FUNCTION: F_RESHAPE},
                                             params.output_shape, params.q_perm_shape[:-1] + (params.d_model,), {}, TFOpLambda,
                                             op_call_args=[(-1,) + params.q_perm_shape[1:-1] + (params.d_model,)], op_call_kwargs={},
                                             functional_op=tf.reshape, **params.reuse_params)
        graph.add_node_with_in_edges(output_reshape_node, [output_node])
        output_permute_node = BaseNode(f'{name}_output_permute', {DIMS: params.output_perm_dims[1:]},
                                       params.q_perm_shape[:-1] + (params.d_model,), params.query_shape[:-1] + (params.d_model,),
                                       {}, Permute, **params.reuse_params)
        graph.add_node_with_in_edges(output_permute_node, [output_reshape_node])

        return output_permute_node

    @staticmethod
    def _connect_to_graph(graph, mha_node, q_permute_node, k_permute_node, v_permute_node, output_permute_node):
        """
        connect subgraph to input graph

        :param graph: input graph
        :param mha_node: MHA node to substitute inputs and outputs with
        :param q_permute_node: 1st input to MHA node
        :param k_permute_node: 2nd input to MHA node
        :param v_permute_node: 3rd input to MHA node
        :param output_permute_node: output node of MHA node
        """
        if len(graph.in_edges(mha_node)) == 3:
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

    def substitute(self,
                   graph: Graph,
                   mha_node: BaseNode) -> Graph:
        """
        Removes a MultiHeadAttention node from the graph, and replaces it with
         a compatible graph that consists of Dense, Dot, Softmax and Concatenate layers.
        Additional reshape and permute nodes are used to shape the inputs to the standard
        of [B, Sequence, C]. All attention axes are folded on the Sequence axis. Iteration
        axes are folded on the Bacth axis

        Args:
            graph: Graph we apply the substitution on.
            mha_node: MultiHeadAttention node to replace.

        Returns:
            Graph after applying the substitution.
        """

        params = MHAParams(mha_node)

        mha_in_edges = graph.in_edges(mha_node)

        # input permutation and reshape to standard shape: (batch, sequence, channels)
        q_permute_node, k_permute_node, v_permute_node, \
        q_reshape_node, k_reshape_node, v_reshape_node = \
            self._standarize_input_shapes(graph, mha_node.name, len(mha_in_edges), params)

        # Generate nodes for attention heads:
        att_head_output_nodes = []
        for head_index in range(params.num_heads):
            dotv_node = self._calc_attention_head(graph, q_reshape_node, k_reshape_node, v_reshape_node,
                                                  mha_node, head_index, params)
            att_head_output_nodes.append(dotv_node)

        # concatenate all attention heads
        concat_node = self._concat_heads(graph, mha_node.name, att_head_output_nodes, params)

        # project output
        output_dense = self._project_output(graph, mha_node, concat_node, params)

        # re-order output to match MHA node output (reshape+permute)
        output_permute_node = self._destandarize_output_shapes(graph, mha_node.name, output_dense, params)

        # connect edges to new nodes
        self._connect_to_graph(graph, mha_node, q_permute_node, k_permute_node, v_permute_node, output_permute_node)

        # Finally, remove the MHA node
        graph.remove_node(mha_node, new_graph_outputs=[OutTensor(output_permute_node, 0)])

        return graph
