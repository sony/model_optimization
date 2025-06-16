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
import copy
import numpy as np
from typing import List, Tuple, Any, Callable

from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import WeightsAttrQuantizationConfig, \
    ActivationQuantizationMode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common import FrameworkInfo, Graph, BaseNode
from model_compression_toolkit.constants import THRESHOLD, SIGNED, SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import create_node_activation_qc, \
    set_quantization_configs_to_node
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_activations_computation \
    import get_activations_qparams
from model_compression_toolkit.core.common.quantization.quantization_params_generation.error_functions import \
    _mse_error_histogram
from model_compression_toolkit.core.common.quantization.quantization_params_generation import z_score_filter
from model_compression_toolkit.target_platform_capabilities import QuantizationMethod, AttributeQuantizationConfig

"""
This substitution aims to solve an issue of activation with negative outputs where
the portion of the negative range is relatively small. In a symmetric quantization this causes 
of bit loosing as the entire negative quantization range does not contain
any values. To solve it, we shift the output of the activation by the minimal output value (quantized) such
that all values after the shifting are positive. To correct the impact of such shifting, a correction
to the next linear node is computed and added to its bias term.
If the linear node pads the input tensor with zeros, we modify the padded value as well.  
"""


def op2d_bias_correction(op2d_node: BaseNode,
                         shift_to_correct: float,
                         bias_str: str,
                         bias_flag_str: str):
    """
    Compute the correction term to add to the op2d node's bias
    to correct the error occurs from adding an Add node (shifting).

    Args:
        op2d_node: Node to compute its bias correction term.
        shift_to_correct: Value that was used to shift the output tensor of
        the non-linear node.
        bias_str:
        bias_flag_str: The framework specific attribute name of the bias flag.
    """

    bias = op2d_node.get_weights_by_keys(bias_str)
    if bias is None:
        bias = 0.0
        op2d_node.framework_attr[bias_flag_str] = True
        # Add an attribute quantization configuration to the newly added bias attribute, with disabled quantization
        for qc in op2d_node.candidates_quantization_cfg:
            qc.weights_quantization_cfg.set_attr_config(bias_flag_str,
                                                        WeightsAttrQuantizationConfig(QuantizationConfig(),
                                                                                      AttributeQuantizationConfig(
                                                                                          enable_weights_quantization=False)))

    # Each node adds a different noise due to the shifting. It depends on the
    # dimensions of the kernel, thus the correction term is a function of
    # the layer type.
    kernel = op2d_node.get_weights_by_keys(op2d_node.kernel_attr)
    if kernel is not None:
        axis_not_output_channel = list(range(len(kernel.shape)))
        axis_not_output_channel.remove(op2d_node.channel_axis.output)

        # special case of depthwise_conv2d in tensorflow, where we have a depth multiplier for the filters
        if op2d_node.channel_axis.output == op2d_node.channel_axis.input:
            axis_not_output_channel.remove(3)  # 3 is the depth multiplier index

        bias_correction = shift_to_correct * np.sum(kernel, axis=tuple(axis_not_output_channel))
        op2d_node.set_weights_by_keys(bias_str, bias - bias_correction.flatten())
    else:
        raise NotImplementedError  # pragma: no cover


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
        Logger.critical(f'Insertion requires exactly one successor node; {len(last_nodes)} successors found.')  # pragma: no cover
    last_node = last_nodes[0]
    insert_node_between_two_nodes(graph, node_to_insert, first_node, last_node)


def insert_node_before_node(graph: Graph,
                            node_to_insert: BaseNode,
                            last_node: BaseNode):
    """
    Insert a new node to a graph before an existing node in the graph.
    Check before insertion that the node (that we add the new node before) has
    only a single incoming edge, so such an insertion is possible. If it is not the
    case, an exception is thrown.

    Args:
        graph: Graph to add the new node to.
        node_to_insert: Node to add.
        last_node: Node to insert the new node after it.

    """
    first_nodes = graph.get_prev_nodes(last_node)
    if len(first_nodes) != 1:
        Logger.critical('Insertion requires exactly one predecessor node; multiple or no predecessors found.')  # pragma: no cover
    first_node = first_nodes[0]
    insert_node_between_two_nodes(graph, node_to_insert, first_node, last_node)


def remove_node_between_two_nodes(graph: Graph,
                                  node_to_remove: BaseNode,
                                  first_node: BaseNode,
                                  last_node: BaseNode):
    """
    Remove a node from a graph and connect its previous node to
    its next node after the removal.

    Args:
        graph: Graph to modify.
        node_to_remove: Node to remove from the graph.
        first_node: Previous node to the node to be removed.
        last_node: Next node to the node to be removed.

    """

    e_attr = graph.get_edge_data(first_node, node_to_remove)
    assert len(list(e_attr.values())) == 1
    e_attr = list(e_attr.values())[0]
    graph.add_edge(first_node, last_node, **e_attr)

    graph.remove_edge(first_node, node_to_remove)
    graph.remove_edge(node_to_remove, last_node)
    graph.remove_node(node_to_remove)


def fuse_padding_with_op2d(graph: 'BaseGraph', pad_node: 'BaseNode', op2d_node: 'BaseNode') -> None:
    """
    Add a padding node to the fused operation containing op2d_node.
    If op2d_node is not already in a fused op, create a new fused group with both nodes.

    Args:
        graph: The computational graph.
        pad_node: The padding node to be added.
        op2d_node: The Conv2D or similar op node following the pad.
    """
    fusing_info = graph.fusing_info

    if fusing_info.is_node_in_fused_op(op2d_node):
        fused_id = fusing_info.get_fused_op_id_for_node(op2d_node.name)
        fused_nodes = fusing_info.get_fused_nodes(fused_id)
        fusing_info.remove_fused_operation(fused_id)
    else:
        fused_nodes = [op2d_node]

    new_fused_nodes = [pad_node] + list(fused_nodes)
    fused_op_id = fusing_info.generate_fused_op_id(new_fused_nodes)

    fusing_info.add_fused_operation(fused_op_id, tuple(new_fused_nodes))
    fusing_info.add_manual_nodes_to_fuse([n.name for n in new_fused_nodes])

def update_fused_op_with_add(graph: 'BaseGraph', non_linear_node: 'BaseNode', add_node: 'BaseNode') -> None:
    """
    Update the fused operation to include an Add node that follows a non-linear activation node.

    Args:
        graph: The computational graph.
        non_linear_node: The non-linear activation node (e.g., ReLU).
        add_node: The Add node inserted after the non-linear node.
    """
    fusing_info = graph.fusing_info
    prev_node = graph.get_prev_nodes(non_linear_node)[0]

    # Gather existing fused nodes (if any)
    fused_candidates = []
    for node in (prev_node, non_linear_node):
        if fusing_info.is_node_in_fused_op(node):
            fused_id = fusing_info.get_fused_op_id_for_node(node.name)
            fused_candidates.extend(fusing_info.get_fused_nodes(fused_id))
    fused_candidates.append(add_node)

    # Remove duplicates while preserving order
    fused_candidates = list(dict.fromkeys(fused_candidates))

    # Remove existing fused ops involving prev_node or non_linear_node
    for node in (prev_node, non_linear_node):
        if fusing_info.is_node_in_fused_op(node):
            fusing_info.remove_fused_operation(fusing_info.get_fused_op_id_for_node(node.name))

    # Register new fused operation
    fused_op_id = fusing_info.generate_fused_op_id(fused_candidates)
    fusing_info.add_manual_nodes_to_fuse([n.name for n in fused_candidates])
    fusing_info.add_fused_operation(fused_op_id, tuple(fused_candidates))


def shift_negative_function(graph: Graph,
                            core_config: CoreConfig,
                            non_linear_node: BaseNode,
                            op2d_node: BaseNode,
                            create_add_node: Callable,
                            get_padding_values: Callable,
                            create_pad_node: Callable,
                            padding_str: str,
                            bias_str: str,
                            bias_flag_str: str,
                            zero_padding_node: BaseNode = None,
                            bypass_nodes: List = None,
                            params_search_quantization_fn: Callable = None
                            ) -> Graph:
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
        core_config: Quantization configuration to build the substitutions list according to.
        non_linear_node: Non-linear node with negative values to shift.
        op2d_node: Linear node to correct its bias to overcome the expected error due to
        the shifting.
        create_add_node: Function to create an add node.
        get_padding_values: Function to compute the op2d node's padding values
        create_pad_node: Function to create an pad node.
        padding_str: The framework specific attribute name of the padding.
        bias_str: The framework specific attribute name of the bias.
        bias_flag_str: The framework specific attribute name of the bias flag.
        zero_padding_node: ZeroPadding2D node that may be in the graph before the linear layer.
        params_search_quantization_fn: Function to quantize np tensor using a framework (tf/torch) quantization method. Needed for better param_search estimating the expected loss.

    Returns:
        Graph after applying the shifting and correction.
    """
    min_to_correct, max_value2compare = graph.get_out_stats_collector(non_linear_node).get_min_max_values()

    if not non_linear_node.is_all_activation_candidates_equal():
        Logger.critical("Shift negative correction is not supported for more than one activation quantization "
                     "configuration candidate")  # pragma: no cover

    # all candidates have same activation config, so taking the first candidate for calculations
    non_linear_node_cfg_candidate = non_linear_node.candidates_quantization_cfg[0].activation_quantization_cfg

    # get the non-linear activation threshold
    activation_threshold = non_linear_node_cfg_candidate.activation_quantization_params.get(THRESHOLD)

    negative_rate = np.abs(min_to_correct) / activation_threshold

    enable_sub = negative_rate <= non_linear_node_cfg_candidate.shift_negative_ratio
    if min_to_correct >= 0 or not enable_sub:
        return graph

    # Calculate the shifting value by checking the quantized points of the shifted activation and
    # taking the minimal quantized point that is still positive.
    num_q_points = 2 ** non_linear_node_cfg_candidate.activation_n_bits
    lsb = activation_threshold / num_q_points
    q_points = np.linspace(0, activation_threshold - lsb, num_q_points).astype(
        'float32')  # Change to type float32 to support tensorflow dtypes

    delta = q_points + min_to_correct
    delta[delta < 0] = np.inf
    shift_value = q_points[np.argmin(delta)]

    if core_config.quantization_config.shift_negative_params_search:

        hist_bins, hist_count = graph.get_out_stats_collector(non_linear_node).hc.get_histogram()
        hist_count = z_score_filter(non_linear_node_cfg_candidate.z_threshold,
                                    hist_bins, hist_count)

        min_mse, _th, _shift = np.inf, None, None
        for _activation_threshold in [activation_threshold, 2 * activation_threshold]:
            qparams = {THRESHOLD: _activation_threshold, SIGNED: False}
            _lsb = _activation_threshold / num_q_points
            _q_points = np.linspace(0, _activation_threshold - _lsb, num_q_points).astype(
                'float32')  # Change to type float32 to support tensorflow dtypes
            for _shift_value in _q_points:
                _hist_bins = hist_bins.astype(np.float32) + _shift_value
                fw_quant_fn = non_linear_node_cfg_candidate.activation_quantization_fn(non_linear_node_cfg_candidate.activation_n_bits,qparams)
                """
                In SNC, when better shifting values are tested for better choice,
                the histogram (which is a numpy object) is quantized using the non-linear node activation
                quantization function (to estimate the expected mse comparing to the original histogram).
                The quantization function is a framework function, which makes it fail since it
                expects a fw tensor. The commmon part of SNC receives an argument which is a callable 
                that receives two argument and returns one: it gets the fw activation quantization function
                and the bins to quantize. The function (of each fw) responsible for doing (if needed) a preprocessing and postprocessing
                to the bins which is a numpy object.
                Only if this function is passed - common SNC will use this function. If not - it assumes
                no processing is needed and simply uses the activation quantization quantizer to quantize the bins (like tf for example).
                """
                if params_search_quantization_fn is None:
                    q_bins = fw_quant_fn(_hist_bins)
                else:
                    q_bins = params_search_quantization_fn(fw_quant_fn, _hist_bins)

                mse = _mse_error_histogram(q_bins, None, _hist_bins, hist_count)
                if mse < min_mse:
                    min_mse = mse
                    _th, _shift = _activation_threshold, _shift_value

        shift_value = _shift
        activation_threshold = _th
        non_linear_node_cfg_candidate.activation_quantization_params[THRESHOLD] = _th

    if zero_padding_node is not None:
        # Remove zero padding layer and save padding values for creating new pad layer
        padding = zero_padding_node.framework_attr.get(padding_str)
        pad_top, pad_btm, pad_left, pad_right = padding[0][0], padding[0][1], padding[1][0], padding[1][1]
        remove_node_between_two_nodes(graph,
                                      node_to_remove=zero_padding_node,
                                      first_node=non_linear_node,
                                      last_node=op2d_node)

    else:
        padding, padding_values = get_padding_values(op2d_node)
        if padding_values is not None:
            pad_top, pad_btm, pad_left, pad_right = padding_values

    # Insert Add node between non linear node to op2d, and fix op2d bias
    add_node = create_add_node(shift_value,
                               non_linear_node.name,
                               non_linear_node.input_shape)

    # Create prior info for the add node
    if non_linear_node.prior_info is not None:
        add_prior_info = non_linear_node.prior_info.get_shifted_prior_info(shift_value)
        add_node.prior_info = add_prior_info

    insert_node_after_node(graph,
                           node_to_insert=add_node,
                           first_node=non_linear_node)
    op2d_bias_correction(op2d_node,
                         shift_value,
                         bias_str,
                         bias_flag_str)

    # Use non linear statistics to create statistics for the Add node according to the shifting
    nl_stats_collector = graph.get_out_stats_collector(non_linear_node)

    add_node_stats_collector = copy.copy(nl_stats_collector)
    graph.set_out_stats_collector_to_node(add_node, add_node_stats_collector)
    graph.shift_stats_collector(add_node, np.array(shift_value))

    set_quantization_configs_to_node(node=add_node,
                                     graph=graph,
                                     quant_config=core_config.quantization_config,
                                     fqc=graph.fqc,
                                     mixed_precision_enable=core_config.is_mixed_precision_enabled)

    update_fused_op_with_add(graph=graph,
                             non_linear_node=non_linear_node,
                             add_node=add_node)

    # If sum([pad_top, pad_btm, pad_left, pad_right])==0 it means we do not pad in any side, thus
    # we do not add a padding node as this is meaningless
    pad_node = None
    if padding is not None and sum([pad_top, pad_btm, pad_left, pad_right])>0:
        pad_node = create_pad_node(op2d_node.name,
                                   add_node.name,
                                   shift_value,
                                   add_node.output_shape,
                                   pad_top, pad_btm, pad_left, pad_right)

        # Insert a pad node between the add node to the op2d, and create statistics for the pad node
        insert_node_before_node(graph,
                                node_to_insert=pad_node,
                                last_node=op2d_node)

        # Set quantization configuration to node, even though we do not quantize it:
        set_quantization_configs_to_node(node=pad_node,
                                         graph=graph,
                                         quant_config=core_config.quantization_config,
                                         fqc=graph.fqc,
                                         mixed_precision_enable=core_config.is_mixed_precision_enabled)

        for candidate_qc in pad_node.candidates_quantization_cfg:
            candidate_qc.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.NO_QUANT
            for attr in pad_node.get_node_weights_attributes():
                candidate_qc.weights_quantization_cfg.get_attr_config(attr).enable_weights_quantization = False

        graph.set_out_stats_collector_to_node(pad_node,
                                              add_node_stats_collector)  # We ignore the padding effect on statistics

        op2d_node.input_shape = pad_node.output_shape

    original_non_linear_activation_nbits = non_linear_node_cfg_candidate.activation_n_bits
    # The non-linear node's output should be float, so we approximate it by using 16bits quantization.
    for candidate_qc in non_linear_node.candidates_quantization_cfg:
        candidate_qc.activation_quantization_cfg.activation_n_bits = SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS

    # A bypass node that has its own activation (e.g. GlobalAvgPool2D) can set it to unsigned
    if bypass_nodes:
        for bypass_node in bypass_nodes:
            for bypass_candidate_qc in bypass_node.candidates_quantization_cfg:
                if bypass_candidate_qc.activation_quantization_cfg:
                    bypass_candidate_qc.activation_quantization_cfg.activation_quantization_params[SIGNED] = False
                    graph.shift_stats_collector(bypass_node, np.array(shift_value))

    add_node_qco = add_node.get_qco(graph.fqc).quantization_configurations
    add_supported_bitwidths = [c.activation_n_bits for c in add_node_qco]
    if original_non_linear_activation_nbits not in add_supported_bitwidths:
        raise ValueError(
            f"Add supported activation bit-widths according to the TPC are: {add_supported_bitwidths}, but non-linear "
            f"bitwidth is {original_non_linear_activation_nbits}. Consider adapting the TPC so 'Add' will support the "
            f"same bitwidth as {non_linear_node.type} or disable shift negative correction.")

    for op_qc_idx, candidate_qc in enumerate(add_node.candidates_quantization_cfg):
        for attr in add_node.get_node_weights_attributes():
            # TODO: do we not quantize the weights of this 'add' on purpose?
            candidate_qc.weights_quantization_cfg.get_attr_config(attr).enable_weights_quantization = False

        candidate_qc.activation_quantization_cfg = create_node_activation_qc(core_config.quantization_config,
                                                                             add_node_qco[op_qc_idx])

        candidate_qc.activation_quantization_cfg.set_activation_quantization_param({THRESHOLD: activation_threshold,
                                                                                    SIGNED: False})

        candidate_qc.activation_quantization_cfg.activation_n_bits = original_non_linear_activation_nbits

    # Add the new padding node to a fused op with the op2d.
    if pad_node:
        fuse_padding_with_op2d(graph=graph,
                               pad_node=pad_node,
                               op2d_node=op2d_node)

    if non_linear_node_cfg_candidate.shift_negative_threshold_recalculation:
        activation_param = get_activations_qparams(activation_quant_cfg=non_linear_node_cfg_candidate,
                                                   nodes_prior_info=non_linear_node.prior_info,
                                                   out_stats_container=graph.get_out_stats_collector(non_linear_node))

        assert activation_param.get(SIGNED) == False
        for candidate_qc in non_linear_node.candidates_quantization_cfg:
            candidate_qc.activation_quantization_cfg.set_activation_quantization_param(activation_param)

    return graph


def get_next_nodes_to_correct(n: BaseNode,
                              graph: Graph,
                              linear_node_types: NodeOperationMatcher,
                              bypass_node_types: NodeOperationMatcher,
                              pad_node_types: NodeOperationMatcher,
                              is_padding_node_and_node_has_padding: Callable,
                              pad_node_to_consider: BaseNode = None,
                              bypass_nodes: List = None) -> Tuple[Any, Any, Any]:
    """
    Search for the next linear node of a given node. Go over
    the next nodes of the node and recursively search for a linear node.

    Args:
        n: Node to search for its next linear node.
        graph: Graph the node is in.
        linear_node_types: Types of linear nodes to consider.
        bypass_node_types: Types of nodes for bypassing to consider.
        pad_node_types: Types of padding nodes to consider.
        is_padding_node_and_node_has_padding: Function to check whether a padding node exists and
         the next node is a linear node with padding.
        pad_node_to_consider: Pad node between the non-linear and linear nodes to consider when
        correcting the expected shift.
        bypass_nodes: a list of bypass nodes found while running this function

    Returns:
        The linear node (if found), a padding node (if found) and a list of bypass nodes (if any), or Nones if it
        were not found or there are multiple outgoing edges to one of nodes during the search (which means, the
        substitution can not be applied).
    """

    next_nodes = graph.get_next_nodes(n)

    if len(next_nodes) != 1:
        return None, None, None

    next_node = next_nodes[0]

    if linear_node_types.apply(next_node):
        # Correction is not supported when there are both padding node and a linear node with padding.
        if is_padding_node_and_node_has_padding(pad_node_to_consider, next_node):
            return None, None, None
        return next_node, pad_node_to_consider, bypass_nodes

    if bypass_node_types.apply(next_node):
        if bypass_nodes:
            bypass_nodes.append(next_node)
        else:
            bypass_nodes = [next_node]
        return get_next_nodes_to_correct(next_node,
                                         graph,
                                         linear_node_types,
                                         bypass_node_types,
                                         pad_node_types,
                                         is_padding_node_and_node_has_padding,
                                         pad_node_to_consider,
                                         bypass_nodes=bypass_nodes)

    if pad_node_types.apply(next_node):
        # Correction is not supported when there are more than one padding node between the non-linear node and the
        # linear node.
        if pad_node_to_consider is None:
            return get_next_nodes_to_correct(next_node,
                                             graph,
                                             linear_node_types,
                                             bypass_node_types,
                                             pad_node_types,
                                             is_padding_node_and_node_has_padding,
                                             next_node,
                                             bypass_nodes=bypass_nodes)

    return None, None, None  # If none of the above were found, it means the correction can not be applied


def apply_shift_negative_correction(graph: Graph,
                                    core_config: CoreConfig,
                                    snc_node_types: NodeOperationMatcher,
                                    linear_node_types: NodeOperationMatcher,
                                    bypass_node_types: NodeOperationMatcher,
                                    pad_node_types: NodeOperationMatcher,
                                    create_add_node: Callable,
                                    get_padding_values: Callable,
                                    create_pad_node: Callable,
                                    is_padding_node_and_node_has_padding: Callable,
                                    padding_str: str,
                                    bias_str: str,
                                    bias_flag_str: str,
                                    params_search_quantization_fn: Callable=None) -> Graph:
    """
    Apply the substitution even if the linear node is not immediately after
    the non-linear node, but there are intermediate nodes

    Args:
        graph: Graph to apply the substitution on.
        core_config: Quantization configuration to build the substitutions list according to.
        groups of layers by how they should be quantized, etc.)
        snc_node_types: Types of activation nodes with negative outputs to consider.
        linear_node_types: Types of linear nodes to consider.
        bypass_node_types: Types of nodes for bypassing to consider.
        pad_node_types: Types of padding nodes to consider.
        create_add_node: Function to create an add node.
        get_padding_values: Function to compute the op2d node's padding values.
        create_pad_node: Function to create an pad node.
        is_padding_node_and_node_has_padding: Function to check whether a padding node exists and
         the next node is a linear node with padding.
        padding_str: The framework specific attribute name of the padding.
        bias_str: The framework specific attribute name of the bias.
        bias_flag_str: The framework specific attribute name of the bias flag.
    Returns:
        Graph after applying shift negative on selected activations.
    """

    nodes = list(graph.nodes())
    for n in nodes:
        # Skip substitution if QuantizationMethod is uniform.
        node_qco = n.get_qco(graph.fqc)
        if any([op_qc.activation_quantization_method is QuantizationMethod.UNIFORM
                for op_qc in node_qco.quantization_configurations]):
            continue

        if snc_node_types.apply(n):
            linear_node, pad_node, bypass_nodes = get_next_nodes_to_correct(n,
                                                                            graph,
                                                                            linear_node_types,
                                                                            bypass_node_types,
                                                                            pad_node_types,
                                                                            is_padding_node_and_node_has_padding
                                                                            )
            if linear_node is not None and n.is_activation_quantization_enabled():
                graph = shift_negative_function(graph,
                                                core_config,
                                                n,
                                                linear_node,
                                                create_add_node,
                                                get_padding_values,
                                                create_pad_node,
                                                padding_str,
                                                bias_str,
                                                bias_flag_str,
                                                zero_padding_node=pad_node,
                                                bypass_nodes=bypass_nodes,
                                                params_search_quantization_fn=params_search_quantization_fn)
    return graph
