# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Tuple
import numpy as np
import tensorflow as tf
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.substitutions.linear_collapsing import Conv2DCollapsing, Op2DAddConstCollapsing
from model_compression_toolkit.core.keras.constants import KERNEL, KERNEL_SIZE, STRIDES, DILATIONS, LINEAR, \
    ACTIVATION, BIAS, USE_BIAS, LAYER_NAME, FILTERS, PADDING, GROUPS, DATA_FORMAT
from model_compression_toolkit.logger import Logger


def linear_collapsing_node_matchers() -> Tuple[NodeOperationMatcher, NodeOperationMatcher]:
    """
    Function generates matchers for matching:
    (Conv2D, Conv2D)[activation=linear] -> Conv2D.
    Returns:
        Matcher for 2 consecutive linear convolution
    """
    first_node = NodeOperationMatcher(Conv2D)
    second_node = NodeOperationMatcher(Conv2D)
    activation_linear = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR)
    first_node = first_node & activation_linear
    return first_node, second_node


def conv2d_collapsing_fn(first_node: BaseNode,
                         second_node: BaseNode,
                         kernel_str: str,
                         bias_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapsing 2 convolutions to one convolution: Out = k2*(k1*x+b1)+b2 = k2*k1*x+k2*b1+b2 = k*x+b
    We calculate k=k2*k1 (collapsed kernel) by injecting identity tensor to the convolutions and extract the output
    We calculate b=k2*b1+b2 (collapsed bias) matrix multiplication
    Args:
        first_node: First layer node to collapse to second layer node
        second_node: Second layer node
        kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
        bias_str: The framework specific attribute name of the convolution layer's bias.
    Returns:
        The modified layer node's weights: kernel, bias
    """
    if first_node.is_match_type(Conv2D) and second_node.is_match_type(Conv2D):
        # Get nodes attributes
        kernel1 = first_node.get_weights_by_keys(kernel_str)
        kernel2 = second_node.get_weights_by_keys(kernel_str)
        bias1 = first_node.get_weights_by_keys(bias_str)
        bias2 = second_node.get_weights_by_keys(bias_str)
        strides1 = first_node.framework_attr[STRIDES]
        strides2 = second_node.framework_attr[STRIDES]

        # --------------------------------------- #
        # Kernel collapsing: k=k2*k1
        # --------------------------------------- #
        # Inspired by https://arxiv.org/pdf/2103.09404.pdf - Algorithm1

        # Generate identity input with padding
        kx, ky = kernel1.shape[0] + kernel2.shape[0] - 1, kernel1.shape[1] + kernel2.shape[1] - 1
        x_pad, y_pad = 2 * kx - 1, 2 * ky - 1
        in_tensor = tf.eye(kernel1.shape[2])
        in_tensor = tf.expand_dims(tf.expand_dims(in_tensor, 1), 1)
        in_tensor = tf.pad(in_tensor, paddings=[[0, 0],
                                                [int(np.ceil((x_pad - 1) / 2)), int(np.floor((x_pad - 1) / 2))],
                                                [int(np.ceil((y_pad - 1) / 2)), int(np.floor((y_pad - 1) / 2))],
                                                [0, 0]])

        # Run first Conv2D
        conv1_out = tf.nn.conv2d(in_tensor, filters=kernel1, strides=strides1, padding="VALID")

        # Run second Conv2D
        conv2_out = tf.nn.conv2d(conv1_out, filters=kernel2, strides=strides2, padding="VALID")

        # Extract collapsed kernel from output: the collapsed kernel is the output of the convolution after fixing the dimension
        kernel_collapsed = tf.transpose(tf.reverse(conv2_out, [1, 2]), [1, 2, 0, 3]).numpy()

        # --------------------------------------- #
        # Bias collapsing: b=k2*b1+b2
        # --------------------------------------- #
        bias_collapsed = None
        if bias1 is not None:
            bias_collapsed = tf.squeeze(tf.matmul(bias1.reshape((1, -1)), tf.math.reduce_sum(kernel2, axis=(0, 1)))).numpy()
            if bias2 is not None:
                bias_collapsed += bias2
        elif bias2 is not None:
            bias_collapsed = bias2

        return kernel_collapsed, bias_collapsed
    else:
        Logger.critical(f"Layer collapsing unsupported for combination: {first_node.type} and {second_node.type}.")  # pragma: no cover


def keras_linear_collapsing() -> Conv2DCollapsing:
    """
    Returns:
        A Conv2DCollapsing initialized for Keras models.
    """
    first_node, second_node = linear_collapsing_node_matchers()
    return Conv2DCollapsing(first_node,
                            second_node,
                            conv2d_collapsing_fn,
                            KERNEL,
                            KERNEL_SIZE,
                            BIAS,
                            USE_BIAS,
                            STRIDES,
                            PADDING,
                            DILATIONS,
                            GROUPS,
                            FILTERS,
                            data_format_str=DATA_FORMAT,
                            layer_name_str=LAYER_NAME)


def op2d_add_const_collapsing_node_matchers() -> Tuple[NodeOperationMatcher, NodeOperationMatcher]:
    """
    Function generates matchers for matching:
    (Op2D, Add(const)) -> Op2D.  (Op2D is one of [DepthwiseConv2D, Conv2D, Conv2DTranspose, Dense)
    Returns:
        Matcher for Op2D followed by Add const
    """
    first_node = NodeOperationMatcher(DepthwiseConv2D) | \
                 NodeOperationMatcher(Conv2D) | \
                 NodeOperationMatcher(Conv2DTranspose) | \
                 NodeOperationMatcher(Dense)
    second_node = NodeOperationMatcher(tf.math.add)
    return first_node, second_node


def op2d_add_const_collapsing_fn(op2d_node: BaseNode,
                                 add_node: BaseNode,
                                 bias_str: str) -> np.ndarray:
    """
    Collapsing Add-Const to previous node's bias
    Args:
        op2d_node: Op2d layer node
        add_node: Add layer to collapse
        bias_str: The framework specific attribute name of the convolution layer's bias.
    Returns:
        The modified conv layer node's bias
    """
    bias = op2d_node.get_weights_by_keys(bias_str)

    # read constant from add node (either 1st or 2nd positional weight)
    const = add_node.weights.get(0, add_node.weights.get(1))
    if const is None:
        Logger.critical(f'Failed to read constant from add node: {add_node.name}.')  # pragma: no cover

    # return new bias
    if bias is None:
        return const
    else:
        return const + bias


def keras_op2d_add_const_collapsing() -> Op2DAddConstCollapsing:
    """
    Returns:
        An Op2DCollapsing initialized for Keras models.
    """
    first_node, second_node = op2d_add_const_collapsing_node_matchers()
    return Op2DAddConstCollapsing(first_node,
                                  second_node,
                                  op2d_add_const_collapsing_fn,
                                  BIAS,
                                  USE_BIAS,
                                  layer_name_str=LAYER_NAME)
