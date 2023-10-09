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
from model_compression_toolkit.core.keras.reader.node_builder import REUSED_IDENTIFIER
from model_compression_toolkit.core.keras.constants import KERNEL, BIAS, USE_BIAS, NUM_HEADS, KEY_DIM, VALUE_DIM, \
    QUERY_SHAPE, KEY_SHAPE, VALUE_SHAPE, OUTPUT_SHAPE, ATTENTION_AXES, ACTIVATION, GROUPS, LINEAR, FILTERS, PADDING, \
    FUNCTION, DIMS, TARGET_SHAPE, F_STRIDED_SLICE, F_STACK, Q_KERNEL, Q_BIAS, K_KERNEL, K_BIAS, V_KERNEL, V_BIAS, \
    OUTPUT_KERNEL, OUTPUT_BIAS, F_MATMUL, TRANSPOSE_B, KERNEL_SIZE, AXIS, F_STRIDED_SLICE_BEGIN, F_STRIDED_SLICE_END, \
    STRIDES, DILATIONS


class DwconvToConv(common.BaseSubstitution):
    """
    Replace DepthWise Conv2d layer into a Conv2d layer with number of groups equal the number of input channel
    """

    def __init__(self):
        """
        Matches DepthwiseConv2D node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(DepthwiseConv2D))

    @staticmethod
    def _get_weight_by_name(mha_node, w_str):
        w = [k for k in mha_node.weights.keys() if w_str in k]
        return w[0] if w else None

    def substitute(self,
                   graph: Graph,
                   dwconv_node: BaseNode) -> Graph:
        """
        Replace DepthwiseConv2D to Conv2D (with groups parameter equals num of input_channels) for regularity of MCT
        quantization parameters.
        Args:
            graph: Graph we apply the substitution on.
            dw_conv_node: DepthwiseConv2D node to replace.
        Returns:
            Graph after applying the substitution.
        """

        # Skip in case mult depth_multiplier=1
        if dwconv_node.framework_attr['depth_multiplier'] == 1:
            return graph

        k = dwconv_node.weights[self._get_weight_by_name(dwconv_node, KERNEL)].copy()
        k_shape = k.shape
        filters = k_shape[2] * k_shape[3]
        k = np.reshape(k,[k_shape[0], k_shape[1], 1, k_shape[2] * k_shape[3]])
        bias_key = self._get_weight_by_name(dwconv_node, BIAS)
        _reuse_params = {REUSE: dwconv_node.reuse, REUSE_GROUP: dwconv_node.reuse_group}

        dwconv_fw_attr = {FILTERS: filters, KERNEL_SIZE: k_shape[0],
                                                    STRIDES: dwconv_node.framework_attr[STRIDES],
                                                    PADDING: dwconv_node.framework_attr[PADDING],
                                                    DILATIONS: dwconv_node.framework_attr[DILATIONS],
                                                    GROUPS: k_shape[2],
                                                    ACTIVATION: dwconv_node.framework_attr[ACTIVATION],
                                                    USE_BIAS: dwconv_node.framework_attr[USE_BIAS]}
        if bias_key:
            b = dwconv_node.weights[bias_key].copy()

            conv_node = BaseNode(dwconv_node.name, dwconv_fw_attr, dwconv_node.input_shape, dwconv_node.output_shape,
                                 {KERNEL: k, BIAS: b}, Conv2D,
                                 **_reuse_params)
        else:
            conv_node = BaseNode(dwconv_node.name, dwconv_fw_attr, dwconv_node.input_shape, dwconv_node.output_shape,
                                 {KERNEL: k}, Conv2D,
                                 **_reuse_params)

        graph.add_node(conv_node)

        # replace DWconv node with Conv node
        _in_edge = list(graph.in_edges(dwconv_node))[0]
        _out_edges = graph.out_edges(dwconv_node)
        graph.add_edge(_in_edge[0], conv_node, **graph.get_edge_data(*_in_edge, 0))
        graph.remove_edge(_in_edge[0], dwconv_node)
        graph.reconnect_out_edges(current_node=dwconv_node, new_node=conv_node)

        # Finally, remove the DWconv node
        graph.remove_node(dwconv_node, new_graph_outputs=[OutTensor(conv_node, 0)])

        return graph