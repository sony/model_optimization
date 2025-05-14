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

from typing import Tuple, Any

import numpy as np

from packaging import version
import tensorflow as tf
from tensorflow.python.keras.layers.core import TFOpLambda
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Activation, Conv2D, Dense, DepthwiseConv2D, ZeroPadding2D, Reshape, \
        GlobalAveragePooling2D, Dropout, ReLU, PReLU, ELU  # pragma: no cover
else:
    from tensorflow.keras.layers import Activation, Conv2D, Dense, DepthwiseConv2D, ZeroPadding2D, Reshape, \
        GlobalAveragePooling2D, Dropout, ReLU, PReLU, ELU

from model_compression_toolkit.core import CoreConfig, FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.substitutions.shift_negative_activation import \
    apply_shift_negative_correction
from model_compression_toolkit.core.keras.constants import KERNEL_SIZE, STRIDES, ACTIVATION, SWISH, \
    SELU, GELU, FUNCTION, ADD, PAD
from model_compression_toolkit.core.keras.constants import NEGATIVE_SLOPE, PADDING, PAD_SAME, PAD_VALID, BIAS, USE_BIAS

SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS = 16

"""
This substitution aims to solve an issue of activation with negative outputs where
the portion of the negative range is relatively small. In a symmetric quantization this causes 
of bit loosing as the entire negative quantization range does not contain
any values. To solve it, we shift the output of the activation by the minimal output value (quantized) such
that all values after the shifting are positive. To correct the impact of such shifting, a correction
to the next linear node is computed and added to its bias term.
If the linear node pads the input tensor with zeros, we modify the padded value as well.  
"""


def shift_negative_activation_node_matchers():
    # Match activation nodes with negative outputs.
    snc_node = NodeOperationMatcher(tf.nn.silu) | \
               NodeOperationMatcher(tf.nn.swish) | \
               NodeOperationMatcher(tf.nn.leaky_relu) | \
               NodeOperationMatcher(tf.nn.selu) | \
               NodeOperationMatcher(tf.nn.gelu) | \
               NodeOperationMatcher(tf.nn.elu) | \
               (NodeOperationMatcher(Activation) & (NodeFrameworkAttrMatcher(ACTIVATION, SWISH) |
                                                    NodeFrameworkAttrMatcher(ACTIVATION, GELU) |
                                                    NodeFrameworkAttrMatcher(ACTIVATION, SELU))) | \
               NodeOperationMatcher(PReLU) | \
               NodeOperationMatcher(ELU) | \
               (NodeOperationMatcher(ReLU) & NodeFrameworkAttrMatcher(NEGATIVE_SLOPE, 0.0).logic_not())  # Leaky ReLU

    # Match linear layers where we can add a correction.
    linear_node = NodeOperationMatcher(Conv2D) | \
                  NodeOperationMatcher(Dense) | \
                  NodeOperationMatcher(DepthwiseConv2D)

    # Match nodes that can be in between the non-linear node to the linear node,
    # and still the substitution can be applied correctly.
    bypass_node = NodeOperationMatcher(Reshape) | \
                  NodeOperationMatcher(GlobalAveragePooling2D) | \
                  NodeOperationMatcher(Dropout)

    # Match a pad node that can be in between the non-linear node to the linear node.
    pad_node = NodeOperationMatcher(ZeroPadding2D)

    return snc_node, linear_node, bypass_node, pad_node


def create_add_node(add_value: float,
                    prev_node_name: str,
                    input_shape: tuple) -> BaseNode:
    """
    Create a new Add node, with a constant to add.
    The name of the node is determined by its previous node's name.

    Args:
        add_value: Constant to add to to the node's input tensor.
        prev_node_name: The name of the node before the Add node
        input_shape: Shape of the Add node's input tensor.

    Returns:
        New Add node.
    """

    add_node_name = prev_node_name + '_post_add'

    add_node = FunctionalNode(add_node_name,
                              {FUNCTION: ADD},
                              input_shape,
                              input_shape,
                              weights={},
                              quantization_attr={},
                              layer_class=TFOpLambda,
                              op_call_args=[np.array(add_value, dtype=np.float32).reshape([1] * len(input_shape))],
                              op_call_kwargs={},
                              functional_op=tf.add)
    return add_node


def create_pad_node(next_node_name: str,
                    prev_node_name: str,
                    value_to_pad: float,
                    input_shape: tuple,
                    pad_top: int,
                    pad_btm: int,
                    pad_left: int,
                    pad_right: int) -> BaseNode:
    """
    Create a pad node with a constant value to pad its input tensor.

    Args:
        next_node_name: Name of the node next to the pad node.
        prev_node_name: Name of the node previous to the pad node.
        value_to_pad: Constant to use for padding the input of the node.
        input_shape: Shape of input tensor.
        pad_top: Number of elements to pad above the tensor.
        pad_btm: Number of elements to pad below the tensor.
        pad_left: Number of elements to pad left to the tensor.
        pad_right: Number of elements to pad right to the tensor.

    Returns:
        A pad node which pads its input with a constant value.
    """
    pad_node_name = next_node_name + '_pre_pad'

    padded_shape = list(input_shape)
    padded_shape[1] += pad_top + pad_btm
    padded_shape[2] += pad_left + pad_right

    num_elements_to_pad = np.array([[0, 0],
                                   [pad_top, pad_btm],
                                   [pad_left, pad_right],
                                   [0, 0]], dtype=np.int32)

    pad_node = FunctionalNode(pad_node_name,
                              {FUNCTION: PAD},
                              input_shape,
                              tuple(padded_shape),
                              weights={},
                              quantization_attr={},
                              layer_class=TFOpLambda,
                              op_call_args=[],
                              op_call_kwargs={'paddings': num_elements_to_pad,
                                              'constant_values': value_to_pad},
                              functional_op=tf.pad)

    return pad_node


def compute_op2d_padding(op2d_node: BaseNode) -> Tuple[int, int, int, int]:
    """
    Compute the padding around an input tensor of a linear node.
    This is needed to replace tensorflow 'same' padding with actual number of elements to pad.

    Args:
        op2d_node: Node to compute the number of elements it adds when padding.

    Returns:
        Tuple of four numbers: number of elements to pad.
    """

    sh = op2d_node.framework_attr.get(STRIDES)[0]
    sw = op2d_node.framework_attr.get(STRIDES)[1]
    kh = op2d_node.framework_attr.get(KERNEL_SIZE)[0]
    kw = op2d_node.framework_attr.get(KERNEL_SIZE)[1]

    pad_along_h = sh * op2d_node.output_shape[1] - op2d_node.input_shape[1] + kh - sh
    pad_along_w = sw * op2d_node.output_shape[2] - op2d_node.input_shape[2] + kw - sw

    pad_top = pad_along_h // 2
    pad_btm = pad_along_h - (pad_along_h // 2)

    pad_left = pad_along_w // 2
    pad_right = pad_along_w - (pad_along_w // 2)

    return pad_top, pad_btm, pad_left, pad_right


def get_padding_values(op2d_node) -> Tuple[Any, Any]:
    """

    Args:
        op2d_node: convolution type node from which to extract the padding values.

    Returns:
        A tuple of containing the padding attribute and padding values.
    """
    padding, padding_values = None, None
    if op2d_node.framework_attr.get(PADDING) == PAD_SAME and not (
            op2d_node.framework_attr.get(KERNEL_SIZE)[0] == 1 and op2d_node.framework_attr.get(KERNEL_SIZE)[1] == 1):
        padding = compute_op2d_padding(op2d_node)
        padding_values = padding[0], padding[1], padding[2], padding[3]
        op2d_node.framework_attr[PADDING] = PAD_VALID
    return padding, padding_values


def is_padding_node_and_node_has_padding(pad_node_to_consider: BaseNode,
                                         next_node: BaseNode) -> bool:
    """

    Args:
        pad_node_to_consider: Pad node between the non-linear and linear nodes to consider when
        correcting the expected shift.
        next_node: The next node after the node in check for correction.

    Returns:
        Whether a padding node exists and the next node is a linear node with padding.
    """
    return pad_node_to_consider is not None and next_node.framework_attr.get(PADDING) == PAD_SAME


def keras_apply_shift_negative_correction(graph: Graph,
                                          core_config: CoreConfig,
                                          fw_info: FrameworkInfo) -> Graph:
    """
    Apply shift negative correction (SNC) on a graph built from a Keras model.

    Args:
        graph: Graph to apply SNC on.
        core_config: Quantization configuration.
        fw_info: FrameworkInfo object with information about the specific framework's module.

    Returns:
        Graph after SNC.
    """
    snc_node, linear_node, bypass_node, pad_node = shift_negative_activation_node_matchers()

    return apply_shift_negative_correction(graph,
                                           core_config,
                                           fw_info,
                                           snc_node,
                                           linear_node,
                                           bypass_node,
                                           pad_node,
                                           create_add_node,
                                           get_padding_values,
                                           create_pad_node,
                                           is_padding_node_and_node_has_padding,
                                           PADDING,
                                           BIAS,
                                           USE_BIAS
                                           )
