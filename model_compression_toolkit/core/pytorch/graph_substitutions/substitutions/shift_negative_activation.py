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
import operator
from typing import Tuple, Any

import torch.nn.functional
from torch.nn import Conv2d, Linear, PReLU, ELU, Hardswish, Dropout, ZeroPad2d, SiLU
from torch import reshape
from torch.nn.functional import hardswish, silu, prelu, elu
from torch.nn.functional import avg_pool2d

from model_compression_toolkit import CoreConfig, FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.graph_matchers import EdgeMatcher
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.substitutions.shift_negative_activation import apply_shift_negative_correction
from model_compression_toolkit.core.pytorch.constants import PAD, VALUE, PADDING, BIAS, USE_BIAS

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
    snc_node = NodeOperationMatcher(PReLU) | \
               NodeOperationMatcher(prelu) | \
               NodeOperationMatcher(ELU) | \
               NodeOperationMatcher(elu) | \
               NodeOperationMatcher(Hardswish) | \
               NodeOperationMatcher(hardswish) | \
               NodeOperationMatcher(SiLU) | \
               NodeOperationMatcher(silu)

    # Match linear layers where we can add a correction.
    linear_node = NodeOperationMatcher(Conv2d) | \
                  NodeOperationMatcher(Linear)

    # Match nodes that can be in between the non-linear node to the linear node,
    # and still the substitution can be applied correctly.
    bypass_node = NodeOperationMatcher(reshape) | \
                  NodeOperationMatcher(avg_pool2d) | \
                  NodeOperationMatcher(Dropout)

    # Match a pad node that can be in between the non-linear node to the linear node.
    pad_node = NodeOperationMatcher(ZeroPad2d)

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

    add_node = common.graph.functional_node.FunctionalNode(add_node_name,
                                                           {},
                                                           input_shape,
                                                           input_shape,
                                                           weights={},
                                                           quantization_attr={},
                                                           functional_op=operator.add,
                                                           op_call_kwargs={},
                                                           op_call_args=[float(add_value)],
                                                           layer_class=operator.add)
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

    op_call_kwargs = {PAD: [pad_left, pad_right, pad_top, pad_btm],
               VALUE: float(value_to_pad)}

    padded_shape = input_shape[0]
    padded_shape[1] += pad_top + pad_btm
    padded_shape[2] += pad_left + pad_right
    pad_node = common.graph.functional_node.FunctionalNode(pad_node_name,
                                                           {},
                                                           input_shape,
                                                           [padded_shape],
                                                           weights={},
                                                           quantization_attr={},
                                                           functional_op=torch.nn.functional.pad,
                                                           op_call_kwargs=op_call_kwargs,
                                                           op_call_args=[],
                                                           layer_class=torch.nn.functional.pad)
    return pad_node


def compute_op2d_padding():
    """
    Dummy function, needed for API.
    Returns:
        None
    """
    return None


def get_padding_values(op2d_node: BaseNode) -> Tuple[Any, Any]:
    """

    Args:
        op2d_node: convolution type node from which to extract the padding values.

    Returns:
        A tuple of containing the padding attribute and padding values.
    """
    padding, padding_values = None, None
    if isinstance(op2d_node.framework_attr.get(PADDING), int):
        padding = op2d_node.framework_attr.get(PADDING)
        padding_values = padding, padding, padding, padding
        op2d_node.framework_attr[PADDING] = 0
    elif isinstance(op2d_node.framework_attr.get(PADDING), tuple):
        padding = op2d_node.framework_attr.get(PADDING)
        padding_values = padding[0], padding[0], padding[1], padding[1]
        op2d_node.framework_attr[PADDING] = 0
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
    padding = next_node.framework_attr.get(PADDING)
    return pad_node_to_consider is not None and (
                (isinstance(padding, int) and padding > 0) or (isinstance(padding, tuple) and sum(padding) > 0))


def pytorch_apply_shift_negative_correction(graph: Graph,
                                            core_config: CoreConfig,
                                            fw_info: FrameworkInfo) -> Graph:
    """
    Apply shift negative correction (SNC) on a graph built from a Pytorch model.

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