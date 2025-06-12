# Copyright 2023 Sony Semiconductors Israel, Inc. All rights reserved.
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
    from keras.src.layers import Dense, Conv2D, Softmax, Concatenate, Reshape, Permute, DepthwiseConv2D
else:
    from keras.layers.core import TFOpLambda
    from keras.layers import Dense, Conv2D, Softmax, Concatenate, Reshape, Permute, DepthwiseConv2D
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.core.keras.constants import KERNEL, BIAS, USE_BIAS, NUM_HEADS, KEY_DIM, VALUE_DIM, \
    QUERY_SHAPE, KEY_SHAPE, VALUE_SHAPE, OUTPUT_SHAPE, ATTENTION_AXES, ACTIVATION, GROUPS, LINEAR, FILTERS, PADDING, \
    FUNCTION, DIMS, TARGET_SHAPE, F_STRIDED_SLICE, F_STACK, Q_KERNEL, Q_BIAS, K_KERNEL, K_BIAS, V_KERNEL, V_BIAS, \
    OUTPUT_KERNEL, OUTPUT_BIAS, F_MATMUL, TRANSPOSE_B, KERNEL_SIZE, AXIS, F_STRIDED_SLICE_BEGIN, F_STRIDED_SLICE_END, \
    DEPTH_MULTIPLIER, DEPTHWISE_INITIALIZER, DEPTHWISE_REGULARIZER, DEPTHWISE_CONSTRAINT, KERNEL_INITIALIZER, \
    KERNEL_REGULARIZER, KERNEL_CONSTRAINT


class DwconvToConv(common.BaseSubstitution):
    """
    A substitution class for replacing DepthwiseConv2D layers with Conv2D layers having 'groups' equal to the number of
    input channels.
    """

    def __init__(self):
        """
        Initializes the DwconvToConv substitution
        """
        super().__init__(matcher_instance=NodeOperationMatcher(DepthwiseConv2D))

    @staticmethod
    def _get_weight_by_name(node, w_str):
        """
        Retrieve the weight with a given name from the node.

        Args:
            node: The node containing weights.
            w_str: The name of the weight to retrieve.

        Returns:
            The weight with the specified name or None if not found.
        """
        w = [k for k in node.weights.keys() if w_str in k]
        return node.weights[w[0]]

    def substitute(self,
                   graph: Graph,
                   dwconv_node: BaseNode) -> Graph:
        """
        Replace a DepthwiseConv2D layer with a Conv2D layer, setting 'groups' parameter to the number of input channels.

        Args:
            graph: The graph on which the substitution is applied.
            dwconv_node: The DepthwiseConv2D node to be replaced.

        Returns:
            The modified graph after applying the substitution.
        """

        # Skip in case mult depth_multiplier=1
        if dwconv_node.framework_attr[DEPTH_MULTIPLIER] == 1:
            return graph

        # Build the new node
        k = self._get_weight_by_name(dwconv_node, KERNEL).copy()
        k_shape = k.shape
        filters = k_shape[2] * k_shape[3] # k_shape[2] * k_shape[3] = number of output channels

        # Transform the DepthwiseConv2D kernel to match the Conv2D kernel, where each input channel is convolved with
        # 'depth_multiplier' filters.
        k = np.reshape(k,[k_shape[0], k_shape[1], 1, filters])
        _reuse_params = {REUSE: dwconv_node.reuse, REUSE_GROUP: dwconv_node.reuse_group}

        conv_fw_attr = dwconv_node.framework_attr
        conv_fw_attr.update({FILTERS: filters,
                               GROUPS: k_shape[2],
                               KERNEL_INITIALIZER: dwconv_node.framework_attr[DEPTHWISE_INITIALIZER],
                               KERNEL_REGULARIZER: dwconv_node.framework_attr[DEPTHWISE_REGULARIZER],
                               KERNEL_CONSTRAINT: dwconv_node.framework_attr[DEPTHWISE_CONSTRAINT]})

        conv_fw_attr.pop(DEPTH_MULTIPLIER)
        conv_fw_attr.pop(DEPTHWISE_INITIALIZER)
        conv_fw_attr.pop(DEPTHWISE_REGULARIZER)
        conv_fw_attr.pop(DEPTHWISE_CONSTRAINT)

        conv_weights = {KERNEL: k}
        if conv_fw_attr[USE_BIAS]:
            b = self._get_weight_by_name(dwconv_node, BIAS).copy()
            conv_weights.update({BIAS: b})

        conv_node = BaseNode(dwconv_node.name, conv_fw_attr, dwconv_node.input_shape, dwconv_node.output_shape,
                             conv_weights, Conv2D,
                             **_reuse_params)

        graph.add_node(conv_node)

        # Replace DWconv node with Conv node
        _in_edge = list(graph.in_edges(dwconv_node))[0]
        _out_edges = graph.out_edges(dwconv_node)
        graph.add_edge(_in_edge[0], conv_node, **graph.get_edge_data(*_in_edge, 0))
        graph.remove_edge(_in_edge[0], dwconv_node)
        graph.reconnect_out_edges(current_node=dwconv_node, new_node=conv_node)

        # Finally, remove the DWconv node
        graph.remove_node(dwconv_node, new_graph_outputs=[OutTensor(conv_node, 0)])

        return graph
