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
import operator
import copy
import numpy as np
from torch.nn import Conv2d, Linear, PReLU, ELU, Hardswish, Dropout, ZeroPad2d
from torch import reshape
from torch.nn.functional import avg_pool2d
from typing import Tuple, Any

from model_compression_toolkit import common
from model_compression_toolkit.common import FrameworkInfo, Graph, BaseNode
from model_compression_toolkit.common.constants import THRESHOLD, SIGNED
from model_compression_toolkit.common.graph.graph_matchers import EdgeMatcher
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.common.quantization.set_node_quantization_config import create_node_activation_qc, \
    set_quantization_configs_to_node
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_activations_computation \
    import \
    get_activations_qparams
from model_compression_toolkit.pytorch.constants import KERNEL, BIAS, SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS

"""
This substitution aims to solve an issue of activation with negative outputs where
the portion of the negative range is relatively small. In a symmetric quantization this causes 
of bit loosing as the entire negative quantization range does not contain
any values. To solve it, we shift the output of the activation by the minimal output value (quantized) such
that all values after the shifting are positive. To correct the impact of such shifting, a correction
to the next linear node is computed and added to its bias term.
If the linear node pads the input tensor with zeros, we modify the padded value as well.  
"""

# Match activation nodes with negative outputs.
SNC_NODE = NodeOperationMatcher(PReLU) | \
           NodeOperationMatcher(ELU) | \
           NodeOperationMatcher(Hardswish)

# Match linear layers where we can add a correction.
LINEAR_NODE = NodeOperationMatcher(Conv2d) | \
              NodeOperationMatcher(Linear)

# Match nodes that can be in between the non-linear node to the linear node,
# and still the substitution can be applied correctly.
BYPASS_NODE = NodeOperationMatcher(reshape) | \
              NodeOperationMatcher(avg_pool2d) | \
              NodeOperationMatcher(Dropout)

# Match a pad node that can be in between the non-linear node to the linear node.
PAD_NODE = NodeOperationMatcher(ZeroPad2d)


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
                                                           op_call_args=[add_value],
                                                           layer_class=operator.add)
    return add_node


def op2d_bias_correction(op2d_node: common.BaseNode,
                         shift_to_correct: float):
    """
    Compute the correction term to add to the op2d node's bias
    to correct the error occurs from adding an Add node (shifting).

    Args:
        op2d_node: Node to compute its bias correction term.
        shift_to_correct: Value that was used to shift the output tensor of
        the non-linear node.

    """

    kernel = op2d_node.get_weights_by_keys(KERNEL)
    bias = op2d_node.get_weights_by_keys(BIAS)
    if bias is None:
        bias = 0.0

    # Each node adds a different noise due to the shifting. It depends on the
    # dimensions of the kernel, thus the correction term is a function of
    # the layer type.
    if op2d_node.layer_class == Conv2d:
        bias_correction = shift_to_correct * np.sum(kernel, axis=(1, 2, 3))
        op2d_node.set_weights_by_keys(BIAS, bias - bias_correction)

    elif op2d_node.layer_class == Linear:
        bias_correction = shift_to_correct * np.sum(kernel, axis=(1))
        op2d_node.set_weights_by_keys(BIAS, bias - bias_correction)

    else:
        raise NotImplementedError


def insert_node_between_two_nodes(graph: Graph,
                                  node_to_insert: BaseNode,
                                  first_node: BaseNode,
                                  last_node: BaseNode):
    """
    Insert a new node in a graph between two nodes.

    Args:
        graph: Graph to add the new node to.
        node_to_insert: Node to add.
        first_node: Node to insert the new node after it.
        last_node: Node to insert the new node before it.

    """

    graph.add_node(node_to_insert)
    e_attr = graph.get_edge_data(first_node, last_node)
    assert len(list(e_attr.values())) == 1
    e_attr = list(e_attr.values())[0]
    graph.add_edge(first_node, node_to_insert, **e_attr)
    graph.add_edge(node_to_insert, last_node, **e_attr)
    graph.remove_edge(first_node, last_node)


def insert_node_after_node(graph: Graph,
                           node_to_insert: BaseNode,
                           first_node: BaseNode):
    """
    Insert a new node to a graph after an existing node in the graph.
    Check before insertion that the node (that we add the new node after) has
    only a single outgoing edge, so such an insertion is possible. If it is not the
    case, an exception is thrown.

    Args:
        graph: Graph to add the new node to.
        node_to_insert: Node to add.
        first_node: Node to insert the new node after it.

    """

    last_nodes = graph.get_next_nodes(first_node)
    if len(last_nodes) != 1:
        raise Exception('Can only insert if there is only one input')
    last_node = last_nodes[0]
    insert_node_between_two_nodes(graph, node_to_insert, first_node, last_node)


def shift_negative_function(graph,
                            qc,
                            non_linear_node,
                            op2d_node,
                            fw_info: FrameworkInfo,
                            zero_padding_node=None):
    """
    Shift the output of a non-linear activation by its minimal output value (quantized) such
    that all values after the shifting are positive.
    The shifting happens only if the ratio between the shifting value and the threshold is small enough
    (the threshold to activate the shifting and correction is in the passed QuantizationConfig, qc).
    To correct the impact of such shifting, a correction to the next linear node is computed and
    added to its bias term.
    If the linear node pads the input tensor with zeros, we modify the padded value as well.

    Args:
        graph: Graph to apply the shifting and correction.
        qc: Quantization configuration to build the substitutions list according to.
        non_linear_node: Non-linear node with negative values to shift.
        op2d_node: Linear node to correct its bias to overcome the expected error due to
        the shifting.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        zero_padding_node: ZeroPadding2D node that may be in the graph before the linear layer.

    Returns:
        Graph after applying the shifting and correction.
    """

    min_to_correct, max_value2compare = graph.get_out_stats_collector(non_linear_node).get_min_max_values()

    # get the non-linear activation threshold
    activation_threshold = non_linear_node.activation_quantization_cfg.activation_quantization_params.get(THRESHOLD)

    negative_rate = np.abs(min_to_correct) / activation_threshold

    enable_sub = negative_rate <= non_linear_node.activation_quantization_cfg.shift_negative_ratio
    if min_to_correct >= 0 or not enable_sub:
        return graph

    # Calculate the shifting value by checking the quantized points of the shifted activation and
    # taking the minimal quantized point that is still positive.
    q_points = np.linspace(0, activation_threshold - activation_threshold / (
            2 ** non_linear_node.activation_quantization_cfg.activation_n_bits),
                           2 ** non_linear_node.activation_quantization_cfg.activation_n_bits).astype(
        'float32')  # Change to type float32 to support tensorflow dtypes

    delta = q_points + min_to_correct
    delta[delta < 0] = np.inf
    shift_value = q_points[np.argmin(delta)]


    # Insert Add node between non linear node to op2d, and fix op2d bias
    add_node = create_add_node(shift_value,
                               non_linear_node.name,
                               non_linear_node.input_shape)
    insert_node_after_node(graph,
                           node_to_insert=add_node,
                           first_node=non_linear_node)
    op2d_bias_correction(op2d_node,
                         shift_value)

    # Use non linear statistics to create statistics for the Add node according to the shifting
    nl_stats_collector = graph.get_out_stats_collector(non_linear_node)

    # The non-linear node's output should be float, so we approximate it by using 16bits quantization.
    non_linear_node.activation_quantization_cfg.activation_n_bits = SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS

    add_node_stats_collector = copy.copy(nl_stats_collector)
    graph.set_out_stats_collector_to_node(add_node, add_node_stats_collector)
    graph.shift_stats_collector(add_node, np.array(shift_value))

    set_quantization_configs_to_node(fw_info=fw_info,
                                     node=add_node,
                                     quant_config=qc)

    add_node.activation_quantization_cfg.enable_activation_quantization = False

    for weight_qc in add_node.candidates_weights_quantization_cfg:
        weight_qc.enable_weights_quantization = False

    add_node.activation_quantization_cfg = create_node_activation_qc(qc,
                                                                     fw_info)

    add_node.activation_quantization_cfg.set_activation_quantization_param({THRESHOLD: activation_threshold,
                                                                            SIGNED: False})

    if non_linear_node.activation_quantization_cfg.shift_negative_threshold_recalculation:
        activation_param = get_activations_qparams(add_node, graph)
        assert activation_param.get(SIGNED) == False
        add_node.activation_quantization_cfg.set_activation_quantization_param(activation_param)

    return graph


def get_next_nodes_to_correct(n: BaseNode,
                              graph: Graph,
                              pad_node_to_consider: BaseNode = None) -> Tuple[Any, Any]:
    """
    Search for the next linear node of a given node. Go over
    the next nodes of the node and recursively search for a linear node.

    Args:
        n: Node to search for its next linear node.
        graph: Graph the node is in.
        pad_node_to_consider: Pad node between the non-linear and linear nodes to consider when
        correcting the expected shift.

    Returns:
        The linear node (if found) and a padding node (if found), or Nones if it were not found or there are
        multiple outgoing edges to one of nodes during the search (which means, the substitution can not be applied).
    """

    next_nodes = graph.get_next_nodes(n)

    if len(next_nodes) != 1:
        return None, None

    next_node = next_nodes[0]

    if LINEAR_NODE.apply(next_node):
        return next_node, pad_node_to_consider

    if BYPASS_NODE.apply(next_node):
        return get_next_nodes_to_correct(next_node, graph, pad_node_to_consider)

    if PAD_NODE.apply(next_node):
        # Correction is not supported when there are more than one padding node between the non-linear node and the
        # linear node.
        if pad_node_to_consider is None:
            return get_next_nodes_to_correct(next_node, graph, next_node)

    return None, None  # If none of the above were found, it means the correction can not be applied


def apply_shift_negative_correction(graph: Graph,
                                    quant_config: QuantizationConfig,
                                    fw_info: FrameworkInfo) -> Graph:
    """
    Apply the substitution even if the linear node is not immediately after
    the non-linear node, but there are intermediate nodes

    Args:
        graph: Graph to apply the substitution on.
        quant_config: Quantization configuration to build the substitutions list according to.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
    Returns:

    """
    nodes = list(graph.nodes())
    for n in nodes:
        if SNC_NODE.apply(n):
            linear_node, pad_node = get_next_nodes_to_correct(n, graph)
            if linear_node is not None:
                graph = shift_negative_function(graph,
                                                quant_config,
                                                n,
                                                linear_node,
                                                fw_info,
                                                zero_padding_node=pad_node)
    return graph
