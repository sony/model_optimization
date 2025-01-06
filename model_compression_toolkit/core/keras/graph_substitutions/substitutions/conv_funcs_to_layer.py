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
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
    from keras.src.layers import Conv2D, DepthwiseConv2D
else:  # pragma: no cover
    from keras.layers.core import TFOpLambda
    from keras.layers import Conv2D, DepthwiseConv2D
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.core.keras.constants import KERNEL, BIAS, USE_BIAS, FILTERS, PADDING, \
    KERNEL_SIZE, DEPTH_MULTIPLIER, STRIDES, DILATIONS, DILATION_RATE, DEPTHWISE_KERNEL, RATE, \
    ACTIVATION, LINEAR, DATA_FORMAT, GROUPS, CHANNELS_FORMAT_FIRST, CHANNELS_FORMAT_LAST


def extract_bias_node_data(_node: FunctionalNode, _graph: Graph) -> np.ndarray:
    """
    Check is can extract bias from next node.

    Args:
        _node: conv node to check for subsequent add\bias_add node to extract bias from.
        _graph: model graph.

    Returns:
        The bias weight. None if couldn't extract bias.

    """
    b = None
    next_nodes = _graph.get_next_nodes(_node)
    if len(next_nodes) == 1 and len(_graph.get_prev_nodes(next_nodes[0])) == 1:
        # Found pattern in graph: conv_node->next_node. Check if next node is add\bias_add that can be absorbed as bias.
        if next_nodes[0].is_match_type(tf.add):
            b = next_nodes[0].weights.get(0, next_nodes[0].weights.get(1))
            if b is not None and len(b.shape) != 1:
                # Constant input to Add node (bias) has irregular shape. Expecting a 1-D array.
                b = None  # pragma: no cover
        elif next_nodes[0].is_match_type(tf.nn.bias_add):
            # In bias_add, weight is always 1-D array. Extract weight from weights or kwargs.
            if 1 in next_nodes[0].weights:
                b = next_nodes[0].weights[1]
            elif BIAS in _node.op_call_kwargs:
                b = np.array(_node.op_call_kwargs[BIAS], dtype=np.float32)

    return b


def replace_conv_node(graph: Graph, new_node: BaseNode, old_node: FunctionalNode, remove_add_node: bool):
    """
    Replace in-place a functional conv node (and possibly subsequent add node) with Conv layer.
    Args:
        graph: model Graph.
        new_node: Conv layer node.
        old_node: conv function node.
        remove_add_node: whether to remove subsequent add node or not.
    """
    graph.add_node(new_node)

    # Replace functional conv node (and potentially add node) with Conv node.
    graph.reconnect_in_edges(old_node, new_node)
    if remove_add_node:
        next_nodes = graph.get_next_nodes(old_node)
        graph.reconnect_out_edges(next_nodes[0], new_node)
        graph.replace_output_node(current_node=next_nodes[0], new_node=new_node)
        graph.remove_edge(old_node, next_nodes[0])
        graph.remove_node(next_nodes[0])
    else:
        graph.reconnect_out_edges(old_node, new_node)
        graph.replace_output_node(current_node=old_node, new_node=new_node)
    graph.remove_node(old_node)


class Conv2dFuncToConv2dLayer(common.BaseSubstitution):
    """
    Substitutes tf.nn.conv2d, tf.compat.v1.nn.conv2d, tf.nn.convolution, tf.compat.v1.nn.convolution functions with a Conv2D layer.
    """

    def __init__(self):
        """
        Initializes the Conv2dFuncToConv2dLayer substitution matcher instance.
        """
        conv2d_matcher = NodeOperationMatcher(tf.nn.conv2d) | NodeOperationMatcher(tf.compat.v1.nn.conv2d)
        convolution_matcher = NodeOperationMatcher(tf.nn.convolution) | NodeOperationMatcher(tf.compat.v1.nn.convolution)
        super().__init__(matcher_instance=conv2d_matcher | convolution_matcher)

    def substitute(self,
                   graph: Graph,
                   conv_func_node: FunctionalNode) -> Graph:
        """
        Substitutes conv functions with a Conv2D layer.

        Args:
            graph: The graph on which the substitution is applied.
            conv_func_node: The functional node to be replaced.

        Returns:
            The modified graph after applying the substitution.
        """

        if 1 in conv_func_node.weights:
            k = conv_func_node.weights[1]
        elif FILTERS in conv_func_node.op_call_kwargs:
            k = np.array(conv_func_node.op_call_kwargs[FILTERS], dtype=np.float32)
        else:
            # Conv weight isn't a constant -> skip substitution.
            return graph  # pragma: no cover

        if len(k.shape) != 4:
            # Conv dimension doesn't match conv2d dimension (K1 x K2 x Cin x Cout) -> skip substitution.
            return graph  # pragma: no cover

        # Check if can extract bias from next node.
        b = extract_bias_node_data(conv_func_node, graph)

        weights = {KERNEL: k}
        # Create Conv2D layer attributes.
        conv_fw_attr = {FILTERS: k.shape[3], KERNEL_SIZE: k.shape[:2], ACTIVATION: LINEAR}
        if len(conv_func_node.op_call_args) > 0:
            Logger.critical(f"node {conv_func_node.name} expected to have only kwargs but got args={conv_func_node.op_call_args}.")  # pragma: no cover

        strides = self._parse_tf_stride_dilation(conv_func_node, STRIDES)
        if strides is None:
            # Non-standard strides -> skip substitution.
            return graph  # pragma: no cover
        conv_fw_attr[STRIDES] = strides

        padding = conv_func_node.op_call_kwargs.get(PADDING) or 'VALID'
        if not isinstance(padding, str):
            # Non-standard padding, Layer only support either 'valid' or 'same' -> skip substitution.
            return graph  # pragma: no cover
        conv_fw_attr[PADDING] = padding

        dilations = self._parse_tf_stride_dilation(conv_func_node, DILATIONS)
        if dilations is None:
            # Non-standard dilations -> skip substitution.
            return graph  # pragma: no cover
        conv_fw_attr[DILATION_RATE] = dilations

        if b is None:
            conv_fw_attr[USE_BIAS] = False
        else:
            weights[BIAS] = b

        data_format = conv_func_node.op_call_kwargs.get(DATA_FORMAT, 'NHWC')
        conv_fw_attr[DATA_FORMAT] = {'NHWC': CHANNELS_FORMAT_LAST, 'NCHW': CHANNELS_FORMAT_FIRST}[data_format]

        conv_fw_attr[GROUPS] = 1

        _reuse_params = {REUSE: conv_func_node.reuse, REUSE_GROUP: conv_func_node.reuse_group}
        conv_node = BaseNode(conv_func_node.name, conv_fw_attr, conv_func_node.input_shape, conv_func_node.output_shape,
                             weights, Conv2D, **_reuse_params)

        replace_conv_node(graph, conv_node, conv_func_node, remove_add_node=b is not None)
        return graph

    def _parse_tf_stride_dilation(self, node, key) -> Optional[Tuple[int, int]]:
        """
        Extract stride/dilation param from tf node and convert it to keras format (suitable for Conv2D).

        Args:
            node: node
            key: param key

        Returns:
            Parsed value or None if non-standard.
        """
        v = node.op_call_kwargs.get(key)
        if v is None:
            return 1, 1
        if isinstance(v, int):
            return v, v
        if len(v) == 1:
            return v[0], v[0]
        if len(v) == 4:
            if v[0] > 1 and v[-1] > 1:
                return None
            else:
                return v[1:3]
        return tuple(v)


class DwConv2dFuncToDwConv2dLayer(common.BaseSubstitution):
    """
    Substitutes tf.nn.depthwise_conv2d & tf.compat.v1.nn.depthwise_conv2d functions with a DepthwiseConv2D layer.
    """

    def __init__(self):
        """
        Initializes the DwConv2dFuncToDwConv2dLayer substitution matcher.
        """
        matcher = NodeOperationMatcher(tf.nn.depthwise_conv2d) | NodeOperationMatcher(tf.compat.v1.nn.depthwise_conv2d)
        super().__init__(matcher_instance=matcher)

    def substitute(self,
                   graph: Graph,
                   dwconv_func_node: FunctionalNode) -> Graph:
        """
        Substitutes dw-conv2d functions with a DepthwiseConv2D layer.

        Args:
            graph: The graph on which the substitution is applied.
            dwconv_func_node: The DepthwiseConv2D node to be replaced.

        Returns:
            The modified graph after applying the substitution.
        """

        if 1 not in dwconv_func_node.weights:
            # Conv weight isn't a constant -> skip substitution.
            return graph  # pragma: no cover

        k = dwconv_func_node.weights[1]

        # Check is can extract bias from next node.
        b = extract_bias_node_data(dwconv_func_node, graph)

        weights = {DEPTHWISE_KERNEL: k}
        k_shape = k.shape
        conv_fw_attr = {DEPTH_MULTIPLIER: k_shape[3], KERNEL_SIZE: k_shape[:2], ACTIVATION: LINEAR}
        if len(dwconv_func_node.op_call_args) > 0:
            Logger.critical(f"node {dwconv_func_node.name} expected to have only kwargs but got args={dwconv_func_node.op_call_args}.")  # pragma: no cover
        if STRIDES in dwconv_func_node.op_call_kwargs:
            strides = dwconv_func_node.op_call_kwargs[STRIDES]
            if strides[0] > 1 or strides[3] > 1:
                # Non-standard strides -> skip substitution.
                return graph  # pragma: no cover
            conv_fw_attr[STRIDES] = strides[1:3]
        if PADDING in dwconv_func_node.op_call_kwargs:
            padding = dwconv_func_node.op_call_kwargs[PADDING]
            if not isinstance(padding, str):
                # Non-standard padding, Layer only support either 'valid' or 'same' -> skip substitution.
                return graph  # pragma: no cover
            conv_fw_attr[PADDING] = padding
        if RATE in dwconv_func_node.op_call_kwargs and dwconv_func_node.op_call_kwargs[RATE] is not None:
            conv_fw_attr[DILATION_RATE] = dwconv_func_node.op_call_kwargs[RATE]
        elif DILATIONS in dwconv_func_node.op_call_kwargs and dwconv_func_node.op_call_kwargs[DILATIONS] is not None:
            conv_fw_attr[DILATION_RATE] = dwconv_func_node.op_call_kwargs[DILATIONS]
        if b is None:
            conv_fw_attr[USE_BIAS] = False
        else:
            weights[BIAS] = b

        _reuse_params = {REUSE: dwconv_func_node.reuse, REUSE_GROUP: dwconv_func_node.reuse_group}
        conv_node = BaseNode(dwconv_func_node.name, conv_fw_attr, dwconv_func_node.input_shape, dwconv_func_node.output_shape,
                             weights, DepthwiseConv2D, **_reuse_params)

        replace_conv_node(graph, conv_node, dwconv_func_node, remove_add_node=b is not None)
        return graph
