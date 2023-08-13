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
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
    from keras.src.layers import MultiHeadAttention, Conv2D, Softmax, Concatenate, Reshape, Permute
else:
    from keras.layers.core import TFOpLambda
    from keras.layers import MultiHeadAttention, Conv2D, Softmax, Concatenate, Reshape, Permute

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.core.keras.constants import KERNEL, BIAS, USE_BIAS, NUM_HEADS, KEY_DIM, VALUE_DIM, \
    QUERY_SHAPE, KEY_SHAPE, VALUE_SHAPE, OUTPUT_SHAPE, ATTENTION_AXES, ACTIVATION, LINEAR, FILTERS, \
    FUNCTION, DIMS, TARGET_SHAPE, F_STRIDED_SLICE, F_STACK, Q_KERNEL, Q_BIAS, K_KERNEL, K_BIAS, V_KERNEL, V_BIAS, \
    OUTPUT_KERNEL, OUTPUT_BIAS, F_MATMUL, KERNEL_SIZE, AXIS, F_STRIDED_SLICE_BEGIN, F_STRIDED_SLICE_END


class MHAParams:
    """
    A data class to hold all relevant parameters from the MHA node framework attributes
    """
    def __init__(self, mha_node):
        """
        Extract MHA params from layer attributes
        Args:
            mha_node: MHA node
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
        iter_axes, self.iter_axes_prod, self.q_att_axes_prod, kv_att_axes_prod = [], 1, 1, 1
        for i, (aq, ak) in enumerate(zip(self.query_shape[1:-1], self.key_shape[1:-1])):
            if i+1 in self.attention_axes:
                self.q_att_axes_prod = self.q_att_axes_prod * aq
                kv_att_axes_prod = kv_att_axes_prod * ak
            else:
                iter_axes.append(i+1)
                self.iter_axes_prod = self.iter_axes_prod * aq
        self.perm_dims = [0] + iter_axes + list(self.attention_axes) + [len(self.query_shape)-1]
        self.output_perm_dims = [self.perm_dims.index(i) for i in range(len(self.perm_dims))]  # dims to revert perm_dims
        self.q_perm_shape = tuple([self.query_shape[i] for i in self.perm_dims])
        self.k_perm_shape = tuple([self.key_shape[i] for i in self.perm_dims])
        self.v_perm_shape = tuple([self.value_shape[i] for i in self.perm_dims])
        self.q_reshape_shape = tuple([self.query_shape[0], self.iter_axes_prod, self.q_att_axes_prod, self.query_shape[-1]])
        self.k_reshape_shape = tuple([self.key_shape[0], self.iter_axes_prod, kv_att_axes_prod, self.key_shape[-1]])
        self.v_reshape_shape = tuple([self.value_shape[0], self.iter_axes_prod, kv_att_axes_prod, self.value_shape[-1]])
        self.query_proj_shape = self.q_reshape_shape[:-1] + (self.query_key_dim,)
        self.key_proj_shape = self.k_reshape_shape[:-1] + (self.query_key_dim,)
        self.value_proj_shape = self.v_reshape_shape[:-1] + (self.value_dim,)
        self.q_slice_shape = self.query_proj_shape[0:1] + self.query_proj_shape[2:]
        self.k_slice_shape = self.key_proj_shape[0:1] + self.key_proj_shape[2:]
        self.v_slice_shape = self.value_proj_shape[0:1] + self.value_proj_shape[2:]
        self.att_matrix_shape = self.q_slice_shape[:-1] + self.k_slice_shape[1:2]
        self.att_output_shape = self.att_matrix_shape[:-1] + self.v_slice_shape[2:3]
        self.stack_shape = self.att_output_shape[:1] + (self.iter_axes_prod,) + self.att_output_shape[1:]
        self.concat_shape = self.stack_shape[:-1] + (self.value_dim*self.num_heads,)
        self.output_shape = self.concat_shape[:-1] + (self.d_model,)
        # self.stacked_output_shape = (self.output_shape[0], self.iter_axes_prod) + self.output_shape[1:]
        self.reuse_params = {REUSE: mha_node.reuse, REUSE_GROUP: mha_node.reuse_group}


class MultiHeadAttentionDecomposition(common.BaseSubstitution):
    """
    Removes a MultiHeadAttention node from the graph,
    and replaces it with a compatible graph that consists of Conv2D,
    tf.matmul, Softmax and Concatenate layers
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
        Add Permute and Reshape nodes to standarize the inputs. output shape will be [B, Iters, Sequence, Channels]
        Args:
            graph: input graph
            name: MHA node name
            num_input_edges: number of input to MHA node. Must be either 2 or 3
            params: MHA params object

        Returns:
            Query, Key & Value nodes to use as inputs to subgraph,
            Standarized Query, Key & Value nodes to connect to the rest of the subgraph

        """
        assert num_input_edges in [2, 3]
        q_permute_node = BaseNode(f'{name}_query_input_permute', {DIMS: params.perm_dims[1:]},
                                  params.query_shape, params.q_perm_shape, {}, Permute, **params.reuse_params)
        graph.add_node(q_permute_node)
        v_permute_node = BaseNode(f'{name}_value_input_permute', {DIMS: params.perm_dims[1:]},
                                  params.value_shape, params.v_perm_shape, {}, Permute, **params.reuse_params)
        graph.add_node(v_permute_node)

        q_reshape_node = BaseNode(f'{name}_query_input_reshape', {TARGET_SHAPE: params.q_reshape_shape[1:]},
                                        params.q_perm_shape, params.q_reshape_shape, {}, Reshape, **params.reuse_params)
        graph.add_node_with_in_edges(q_reshape_node, [q_permute_node])
        v_reshape_node = BaseNode(f'{name}_value_input_reshape', {TARGET_SHAPE: params.v_reshape_shape[1:]},
                                        params.v_perm_shape, params.v_reshape_shape, {}, Reshape, **params.reuse_params)
        graph.add_node_with_in_edges(v_reshape_node, [v_permute_node])
        if num_input_edges == 3:
            # MHA node is called with 3 inputs: Query, Value & Key
            k_permute_node = BaseNode(f'{name}_key_input_permute', {DIMS: params.perm_dims[1:]},
                                      params.key_shape, params.k_perm_shape, {}, Permute, **params.reuse_params)
            graph.add_node(k_permute_node)
            k_reshape_node = BaseNode(f'{name}_key_input_reshape', {TARGET_SHAPE: params.k_reshape_shape[1:]},
                                            params.k_perm_shape, params.k_reshape_shape, {}, Reshape, **params.reuse_params)
            graph.add_node_with_in_edges(k_reshape_node, [k_permute_node])
        else:
            # MHA node is called with 2 inputs: Query & Value. Key=Value
            k_permute_node = v_permute_node
            k_reshape_node = v_reshape_node

        return (q_permute_node, k_permute_node, v_permute_node,
                q_reshape_node, k_reshape_node, v_reshape_node)

    @staticmethod
    def _slice_per_iteration(graph, name, head_index, i_iter, q_node, k_node, v_node, params):
        """
        Prepare MHA data for attention: Slice inputs on Iters axis.
         [B, Iters, Sequence, Channels] --> [B, Sequence, Channels]

        Args:
            graph: input graph
            name: MHA node name
            head_index: head index
            i_iter: iteration index, from 0..Iters-1
            q_node: query input after projection
            k_node: key input after projection
            v_node: value input after projection
            params: MHA params object

        Returns:
            inputs to MHA node, per iteration
        """
        _shape = params.query_proj_shape
        q_slice_node_no_shrink = FunctionalNode(f'{name}_q{head_index}_slice{i_iter}', {FUNCTION: F_STRIDED_SLICE},
                                                _shape, (_shape[0], 1) + _shape[2:], {}, TFOpLambda,
                                                op_call_args=[[0, i_iter, 0, 0], [0, i_iter + 1, 0, 0]],
                                                op_call_kwargs={F_STRIDED_SLICE_BEGIN: 13, F_STRIDED_SLICE_END: 13},
                                                functional_op=tf.strided_slice, **params.reuse_params)
        graph.add_node_with_in_edges(q_slice_node_no_shrink, [q_node])
        _shape = params.key_proj_shape
        k_slice_node_no_shrink = FunctionalNode(f'{name}_k{head_index}_slice{i_iter}', {FUNCTION: F_STRIDED_SLICE},
                                                _shape, (_shape[0], 1) + _shape[2:], {}, TFOpLambda,
                                                op_call_args=[[0, i_iter, 0, 0], [0, i_iter + 1, 0, 0]],
                                                op_call_kwargs={F_STRIDED_SLICE_BEGIN: 13, F_STRIDED_SLICE_END: 13},
                                                functional_op=tf.strided_slice, **params.reuse_params)
        graph.add_node_with_in_edges(k_slice_node_no_shrink, [k_node])
        _shape = params.value_proj_shape
        v_slice_node_no_shrink = FunctionalNode(f'{name}_v{head_index}_slice{i_iter}', {FUNCTION: F_STRIDED_SLICE},
                                                _shape, (_shape[0], 1) + _shape[2:], {}, TFOpLambda,
                                                op_call_args=[[0, i_iter, 0, 0], [0, i_iter + 1, 0, 0]],
                                                op_call_kwargs={F_STRIDED_SLICE_BEGIN: 13, F_STRIDED_SLICE_END: 13},
                                                functional_op=tf.strided_slice, **params.reuse_params)
        graph.add_node_with_in_edges(v_slice_node_no_shrink, [v_node])

        q_slice_node = BaseNode(f'{name}_q{head_index}_slice_reshape{i_iter}', {TARGET_SHAPE: params.q_slice_shape[1:]},
                                q_slice_node_no_shrink.output_shape, params.q_slice_shape, {},
                                Reshape, **params.reuse_params)
        graph.add_node_with_in_edges(q_slice_node, [q_slice_node_no_shrink])
        k_slice_node = BaseNode(f'{name}_k{head_index}_slice_reshape{i_iter}', {TARGET_SHAPE: params.k_slice_shape[1:]},
                                k_slice_node_no_shrink.output_shape, params.k_slice_shape, {},
                                Reshape, **params.reuse_params)
        graph.add_node_with_in_edges(k_slice_node, [k_slice_node_no_shrink])
        v_slice_node = BaseNode(f'{name}_v{head_index}_slice_reshape{i_iter}', {TARGET_SHAPE: params.v_slice_shape[1:]},
                                v_slice_node_no_shrink.output_shape, params.v_slice_shape, {},
                                Reshape, **params.reuse_params)
        graph.add_node_with_in_edges(v_slice_node, [v_slice_node_no_shrink])

        return q_slice_node, k_slice_node, v_slice_node

    def _project_inputs(self, graph, mha_node, head_index, q_reshape_node, k_reshape_node, v_reshape_node, params):
        """
        Create projection nodes (as Conv2D 1x1)

        Args:
            graph: input graph
            mha_node: MHA node name
            head_index: head index
            q_reshape_node: query input after standardization
            k_reshape_node: key input after standardization
            v_reshape_node: value input after standardization
            params: MHA params object

         Returns:
            Projection nodes

        """
        # add norm factor to query kernel and bias
        factor = (params.query_key_dim ** -0.5)
        qk = mha_node.weights[self._get_weight_by_name(mha_node, Q_KERNEL)][:, head_index, :].copy() * factor
        kk = mha_node.weights[self._get_weight_by_name(mha_node, K_KERNEL)][:, head_index, :].copy()
        vk = mha_node.weights[self._get_weight_by_name(mha_node, V_KERNEL)][:, head_index, :].copy()
        qb = mha_node.weights[self._get_weight_by_name(mha_node, Q_BIAS)][head_index, :].copy() * factor
        kb = mha_node.weights[self._get_weight_by_name(mha_node, K_BIAS)][head_index, :].copy()
        vb = mha_node.weights[self._get_weight_by_name(mha_node, V_BIAS)][head_index, :].copy()
        qk = qk[np.newaxis, np.newaxis, ...]
        kk = kk[np.newaxis, np.newaxis, ...]
        vk = vk[np.newaxis, np.newaxis, ...]

        # project query, key & value inputs to query_key_dim, query_key_dim & value_dim respectively
        query_name = f'{mha_node.name}_query_{head_index}'
        q_node = BaseNode(query_name, {FILTERS: params.query_key_dim, KERNEL_SIZE: 1, USE_BIAS: params.use_bias,
                                       ACTIVATION: LINEAR},
                          params.q_reshape_shape, params.query_proj_shape, {KERNEL: qk, BIAS: qb}, Conv2D,
                          **params.reuse_params)
        graph.add_node_with_in_edges(q_node, [q_reshape_node])
        key_name = f'{mha_node.name}_key_{head_index}'
        k_node = BaseNode(key_name, {FILTERS: params.query_key_dim, KERNEL_SIZE: 1, USE_BIAS: params.use_bias, ACTIVATION: LINEAR},
                          params.k_reshape_shape, params.key_proj_shape, {KERNEL: kk, BIAS: kb}, Conv2D,
                          **params.reuse_params)
        graph.add_node_with_in_edges(k_node, [k_reshape_node])
        value_name = f'{mha_node.name}_value_{head_index}'
        v_node = BaseNode(value_name, {FILTERS: params.value_dim, KERNEL_SIZE: 1, USE_BIAS: params.use_bias, ACTIVATION: LINEAR},
                          params.v_reshape_shape, params.value_proj_shape, {KERNEL: vk, BIAS: vb}, Conv2D,
                          **params.reuse_params)
        graph.add_node_with_in_edges(v_node, [v_reshape_node])
        return q_node, k_node, v_node

    @staticmethod
    def _calc_attention_head(graph, q_slice_node, k_slice_node, v_slice_node, mha_node,
                             iter_index, head_index, params):
        """
        Generate the attention head subgraph: matmul(softmax(matmul(projected_Q, projected_K)), projected_V)

        Args:
            graph: input graph
            q_slice_node: input query node
            k_slice_node: input key node
            v_slice_node: input value node
            mha_node: MHA node
            iter_index: iteration index
            head_index: head index being generated. used to set correct node names
            params: MHA params object

        Returns:
            output of attention head

        """

        # calculate attention matrix:
        _k_transposed_shape = (params.k_slice_shape[0], params.k_slice_shape[2], params.k_slice_shape[1])
        matmul_pre_transpose_node = BaseNode(f'{mha_node.name}_qk{head_index}_matmul{iter_index}_pre_transpose', {DIMS: [2, 1]},
                                             params.k_slice_shape, _k_transposed_shape, {}, Permute, **params.reuse_params)
        graph.add_node_with_in_edges(matmul_pre_transpose_node, [k_slice_node])

        matmul_node = FunctionalNode(f'{mha_node.name}_qk{head_index}_matmul{iter_index}', {FUNCTION: F_MATMUL},
                                     (params.q_slice_shape, _k_transposed_shape), params.att_matrix_shape, {},
                                     TFOpLambda, op_call_args=[], op_call_kwargs={},
                                     functional_op=tf.matmul, **params.reuse_params)
        graph.add_node_with_in_edges(matmul_node, [q_slice_node, matmul_pre_transpose_node])

        # apply softmax on attention matrix
        softmax_name = f'{mha_node.name}_softmax{head_index}_{iter_index}'
        softmax_node = BaseNode(softmax_name, {},
                                params.att_matrix_shape, params.att_matrix_shape,
                                {}, Softmax, **params.reuse_params)
        graph.add_node_with_in_edges(softmax_node, [matmul_node])

        # multiply attention matrix with projected values
        matmulv_node = FunctionalNode(f'{mha_node.name}_v{head_index}_matmul{iter_index}', {FUNCTION: F_MATMUL},
                                      (params.att_matrix_shape, params.v_slice_shape), params.att_output_shape,
                                      {}, TFOpLambda, op_call_args=[], op_call_kwargs={},
                                      functional_op=tf.matmul, **params.reuse_params)
        graph.add_node_with_in_edges(matmulv_node, [softmax_node, v_slice_node])
        return matmulv_node

    @staticmethod
    def _stack_iters(graph, name, input_nodes, head_index, params):
        """
        Stack iteration slices before output projection
         [B, Sequence, Channels] x Iters --> [B, Iters, Sequence, Channels]

        Args:
            graph: input graph
            name: MHA node name
            input_nodes: input nodes list
            head_index: current head index
            params: MHA params object

        Returns:
            stacked outputs node

        """
        if params.iter_axes_prod == 1:
            output_stacked = BaseNode(f'{name}_stack{head_index}_as_reshape', {TARGET_SHAPE: params.stack_shape[1:]},
                                      input_nodes[0].output_shape, params.stack_shape, {},
                                      Reshape, **params.reuse_params)
        else:
            output_stacked = FunctionalNode(f'{name}_stack{head_index}', {FUNCTION: F_STACK},
                                            tuple([n.output_shape for n in input_nodes]), params.stack_shape, {},
                                            TFOpLambda, op_call_args=[], op_call_kwargs={AXIS: 1},
                                            functional_op=tf.stack, inputs_as_list=True, **params.reuse_params)
        graph.add_node_with_in_edges(output_stacked, input_nodes)
        return output_stacked

    @staticmethod
    def _concat_heads(graph, name, input_nodes, params):
        """
        concatenate attention heads
        Args:
            graph: input graph
            name: MHA node name
            input_nodes: all the outputs of the attention heads
            params: MHA params object

        Returns: concat node

        """
        concat_node = BaseNode(f'{name}_concat_heads', {},
                               tuple([n.output_shape for n in input_nodes]), params.concat_shape, {}, Concatenate,
                               **params.reuse_params)
        graph.add_node_with_in_edges(concat_node, input_nodes)

        return concat_node

    def _project_output(self, graph, mha_node, input_node, params):
        """
        project all attention outputs (after concatenation)
        Args:
            graph: input graph
            mha_node: MHA node
            input_node: concat node
            params: MHA params object

        Returns:
            output node of the MHA node
        """
        w_out = mha_node.weights[self._get_weight_by_name(mha_node, OUTPUT_KERNEL)].copy().reshape((1, 1, -1, params.d_model))
        b_out = mha_node.weights[self._get_weight_by_name(mha_node, OUTPUT_BIAS)].copy()
        proj_output_name = f'{mha_node.name}_output_conv1x1'
        output = BaseNode(proj_output_name, {FILTERS: params.d_model, KERNEL_SIZE: 1, USE_BIAS: params.use_bias, ACTIVATION: LINEAR},
                          params.concat_shape, params.output_shape, {KERNEL: w_out, BIAS: b_out}, Conv2D,
                          **params.reuse_params)
        graph.add_node_with_in_edges(output, [input_node])
        return output

    @staticmethod
    def _destandarize_output_shapes(graph, name, output_node, params):
        """
        return output to original MHA output shape
        Args:
            graph: input graph
            name: MHA node name
            output_node: output node of output projection
            params: MHA params object

        Returns:
            Destandarized output node, which matches in shape the output of the MHA node
        """

        output_reshape_node = BaseNode(f'{name}_output_reshape', {TARGET_SHAPE: params.q_perm_shape[1:-1] + (params.d_model,)},
                                       params.output_shape, params.q_perm_shape[:-1] + (params.d_model,), {},
                                       Reshape, **params.reuse_params)
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
        Args:
            graph: input graph
            mha_node: MHA node to substitute inputs and outputs with
            q_permute_node: 1st input to MHA node
            k_permute_node: 2nd input to MHA node
            v_permute_node: 3rd input to MHA node
            output_permute_node: output node of MHA node

        Returns:

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
         a compatible graph that consists of Dense, strided_slice, Dot, Softmax and Concatenate layers.
        Additional reshape and permute nodes are used to shape the inputs to the standard
        of [B, Iters, Sequence, C]. All attention axes are folded on the Sequence axis and iteration axes
        to the Iters axis.

        Args:
            graph: Graph we apply the substitution on.
            mha_node: MultiHeadAttention node to replace.

        Returns:
            Graph after applying the substitution.
        """

        if mha_node.reuse:
            Logger.error("MCT doesn't support reuse of MultiHeadAttention layer")  # pragma: no cover
        params = MHAParams(mha_node)

        mha_in_edges = graph.in_edges(mha_node)

        # input permutation and reshape to standard shape: (batch, iterations, sequence, channels)
        q_permute_node, k_permute_node, v_permute_node, \
        q_reshape_node, k_reshape_node, v_reshape_node = \
            self._standarize_input_shapes(graph, mha_node.name, len(mha_in_edges), params)

        head_outputs = []
        for head_index in range(params.num_heads):
            q_node, k_node, v_node = self._project_inputs(graph, mha_node, head_index,
                                                          q_reshape_node, k_reshape_node, v_reshape_node, params)

            att_outputs = []
            for i_iter in range(params.iter_axes_prod):
                q_slice_node, k_slice_node, v_slice_node = self._slice_per_iteration(graph, mha_node.name, head_index, i_iter,
                                                                                     q_node, k_node, v_node,
                                                                                     params)

                matmulv_node = self._calc_attention_head(graph, q_slice_node, k_slice_node, v_slice_node, mha_node,
                                                         i_iter, head_index, params)

                att_outputs.append(matmulv_node)

            output_stacked = self._stack_iters(graph, mha_node.name, att_outputs, head_index, params)
            head_outputs.append(output_stacked)

        concat_node = self._concat_heads(graph, mha_node.name, head_outputs, params)

        output = self._project_output(graph, mha_node, concat_node, params)

        # re-order output to match MHA node output (reshape+permute)
        output_permute_node = self._destandarize_output_shapes(graph, mha_node.name, output, params)

        # connect edges to new nodes
        self._connect_to_graph(graph, mha_node, q_permute_node, k_permute_node, v_permute_node, output_permute_node)

        # Finally, remove the MHA node
        graph.remove_node(mha_node, new_graph_outputs=[OutTensor(output_permute_node, 0)])

        return graph
