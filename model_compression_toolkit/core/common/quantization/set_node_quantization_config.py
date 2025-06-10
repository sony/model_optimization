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
from typing import List, Tuple

from model_compression_toolkit.core import BitWidthConfig
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.framework_info import get_fw_info, ChannelAxisMapping
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig, TPCQuantizationInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.core.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import max_input_activation_n_bits
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities
from model_compression_toolkit.logger import Logger


def set_quantization_configuration_to_graph(graph: Graph,
                                            fqc: FrameworkQuantizationCapabilities):
    """
    Add quantization configuration for each graph node.

    Args:
        graph (Graph): Graph for which to add quantization info to each node.
        fqc: framework quantization capabilities.

    Returns:
        Graph: The graph with quantization configurations attached to each node in it.
    """

    for n in graph.get_topo_sorted_nodes():
        set_quantization_configs_to_node(node=n,
                                         graph=graph,
                                         fqc=fqc)
    return graph


def filter_node_qco_by_graph(node: BaseNode,
                             fqc: FrameworkQuantizationCapabilities,
                             graph: Graph,
                             node_qc_options: QuantizationConfigOptions
                             ) -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
    """
    Filter quantization config options that don't match the graph.
    A node may have several quantization config options with 'activation_n_bits' values, and
    the next nodes in the graph may support different bit-width as input activation. This function
    filters out quantization config that don't comply to these attributes.

    Args:
        node: Node for filtering.
        fqc: FQC to extract the QuantizationConfigOptions for the next nodes.
        graph: Graph object.
        node_qc_options: Node's QuantizationConfigOptions.

    Returns:
        A base config (OpQuantizationConfig) and a config options list (list of OpQuantizationConfig)
        that are compatible with next nodes supported input bit-widths.

    """
    # Filter quantization config options that don't match the graph.
    _base_config = node_qc_options.base_config
    _node_qc_options = node_qc_options.quantization_configurations

    # Build next_nodes list by appending to the node's next nodes list all nodes that are quantization preserving.
    _next_nodes = graph.get_next_nodes(node)
    next_nodes = []
    while len(_next_nodes):
        n = _next_nodes.pop(0)
        qco = n.get_qco(fqc)
        qp = [qc.quantization_preserving for qc in qco.quantization_configurations]
        if not all(qp) and any(qp):
            Logger.error(f'Attribute "quantization_preserving" should be the same for all QuantizaionConfigOptions in {n}.')
        if qp[0]:
            _next_nodes.extend(graph.get_next_nodes(n))
        next_nodes.append(n)

    if len(next_nodes) == 0:
        return _base_config, _node_qc_options
    next_nodes_qc_options = [_node.get_qco(fqc) for _node in next_nodes]
    all_next_nodes_supported_input_bitwidth = [max_input_activation_n_bits(op_cfg)
                                                   for qc_opts in next_nodes_qc_options
                                                   for op_cfg in qc_opts.quantization_configurations
                                               if op_cfg.enable_activation_quantization or op_cfg.quantization_preserving
                                               ]
    if len(all_next_nodes_supported_input_bitwidth):
        next_nodes_supported_input_bitwidth = min(all_next_nodes_supported_input_bitwidth)

        # Filter node's QC options that match next nodes input bit-width.
        _node_qc_options = [_option for _option in _node_qc_options
                            if _option.activation_n_bits <= next_nodes_supported_input_bitwidth]
        if len(_node_qc_options) == 0:
            Logger.critical(f"Graph doesn't match FQC bit configurations: {node} -> {next_nodes}.")

        # Verify base config match
        if any([node_qc_options.base_config.activation_n_bits > max_input_activation_n_bits(qc_opt.base_config)
                for qc_opt in next_nodes_qc_options]):
            # base_config activation bits doesn't match next node supported input bit-width -> replace with
            # a qco from quantization_configurations with maximum activation bit-width.
            if len(_node_qc_options) > 0:
                output_act_bitwidth = {qco.activation_n_bits: i for i, qco in enumerate(_node_qc_options)}
                _base_config = _node_qc_options[output_act_bitwidth[max(output_act_bitwidth)]]
                Logger.warning(f"Node {node} base quantization config changed to match Graph and FQC configuration.\nCause: {node} -> {next_nodes}.")
            else:
                Logger.critical(f"Graph doesn't match FQC bit configurations: {node} -> {next_nodes}.")  # pragma: no cover

    return _base_config, _node_qc_options


def set_quantization_configs_to_node(node: BaseNode,
                                     graph: Graph,
                                     fqc: FrameworkQuantizationCapabilities):
    """
    Create and set quantization configurations to a node (for both weights and activation).

    Args:
        node (BaseNode): Node to set its quantization configurations.
        graph (Graph): Model's internal representation graph.
        fqc (FrameworkQuantizationCapabilities): FrameworkQuantizationCapabilities to get default OpQuantizationConfig.
    """
    node_qc_options = node.get_qco(fqc)
    base_config, node_qc_options_list = filter_node_qco_by_graph(node, fqc, graph, node_qc_options)

    # Create QC candidates for weights and activation combined
    weight_channel_axis = fw_info.kernel_channels_mapping.get(node.type)

    node_attrs_list = node.get_node_weights_attributes()
    mp_candidates = [_create_node_single_candidate_qc(fw_info, weight_channel_axis, op_cfg, node_attrs_list)
                     for op_cfg in node_qc_options_list]
    sp_cfg = _create_node_single_candidate_qc(fw_info, weight_channel_axis, base_config, node_attrs_list)
    node.tpc_quantization_info = TPCQuantizationInfo(base_quantization_cfg=sp_cfg,
                                                     candidates_quantization_cfg=mp_candidates)
    node.sort_node_candidates(fw_info)
    if not node.get_has_activation():
        node.tpc_quantization_info.update_activation_quantization_mode(ActivationQuantizationMode.NO_QUANT)


    if candidate_qc.activation_quantization_cfg.quant_mode == ActivationQuantizationMode.PRESERVE_QUANT:
            prev_nodes = graph.get_prev_nodes(node)
            if len(prev_nodes) != 1:
                # Preserving the quantization of more than 1 previous node is ambiguous, so disable it.
                Logger.info(f"Disabling Quantization-Preserving for node {node.name} because it has more than 1 input activations.")
                candidate_qc.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.NO_QUANT
            elif not prev_nodes[0].is_quantization_preserving() and not prev_nodes[0].is_activation_quantization_enabled():
                # Preserving the quantization of an unquantized node isn't possible, so disable it.
                Logger.info(f"Disabling Quantization-Preserving for node {node.name} because previous node activation quantization is disabled.")
                candidate_qc.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.NO_QUANT


def create_node_activation_qc(op_cfg: OpQuantizationConfig) -> NodeActivationQuantizationConfig:
    """
    Create an activation quantization configuration from a QuantizationConfig object.

    Args:
        weights/activations should be quantized)
        op_cfg: OpQuantizationConfig with quantizers types to set in node quantization configuration.

    Returns:
        Activation quantization configuration of a node.
    """

    activation_quantization_fn = get_fw_info().activation_quantizer_mapping.get(op_cfg.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown activation quantization method specified.')  # pragma: no cover

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method)

    return NodeActivationQuantizationConfig(op_cfg,
                                            activation_quantization_fn,
                                            activation_quantization_params_fn)


def _create_node_single_candidate_qc(weight_channel_axis: Tuple[int, int],
                                     op_cfg: OpQuantizationConfig,
                                     node_attrs_list: List[str]) -> CandidateNodeQuantizationConfig:
    """
    Create quantization configuration candidate from a QuantizationConfig object.
    Creates both weights and activation quantization configurations
    and initialize a candidate object that encapsulates both.

    Args:
        weight_channel_axis: (Output, Input) channel index of the node's kernel.
        op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
        node_attrs_list: A list of the node's weights attributes names.

    Returns: a CandidateNodeQuantizationConfig object with both weights and activation quantization config objects.

    """

    # parameters for weights attributes quantization are set within  CandidateNodeQuantizationConfig initialization

    # get parameters for activation quantization
    activation_quantization_fn = get_fw_info().activation_quantizer_mapping.get(op_cfg.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown activation quantization method specified.')  # pragma: no cover

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method)

    # TODO: remove this validation and warning once enabling all attributes quantization by default
    attrs_with_enabled_quantization = [attr for attr, cfg in op_cfg.attr_weights_configs_mapping.items()
                                       if cfg.enable_weights_quantization]
    if len(attrs_with_enabled_quantization) > 1:
        Logger.warning(f"Multiple weights attributes quantization is enabled via the provided FQC."
                       f"Quantizing any attribute other than the kernel is experimental "
                       f"and may be subject to unstable behavior."
                       f"Attributes with enabled weights quantization: {attrs_with_enabled_quantization}.")

    return CandidateNodeQuantizationConfig(op_cfg=op_cfg,
                                           activation_quantization_fn=activation_quantization_fn,
                                           activation_quantization_params_fn=activation_quantization_params_fn,
                                           weights_channels_axis=weight_channel_axis,
                                           node_attrs_list=node_attrs_list)


def _create_node_candidates_qc(weight_channel_axis: Tuple[int, int],
                               node_qc_options_list: List[OpQuantizationConfig],
                               base_config: OpQuantizationConfig,
                               node: BaseNode) -> Tuple[List[CandidateNodeQuantizationConfig], CandidateNodeQuantizationConfig]:

    """
    Create a list of candidates of weights and activation quantization configurations for a node.

    Args:
        weight_channel_axis (Tuple[int, int]): (Output, Input) channel index of the node's kernel.
        node_qc_options_list (List[OpQuantizationConfig]): List of quantization configs of node.
        base_config (OpQuantizationConfig): Base quantization config for node.
        node (BaseNode): A node to set quantization configuration candidates to.

    Returns:
        List[CandidateNodeQuantizationConfig]: List of candidates of weights quantization configurations to set for a node.
    """

    node_attrs_list = node.get_node_weights_attributes()
    candidates = [_create_node_single_candidate_qc(fw_info, weight_channel_axis, op_cfg, node_attrs_list)
                  for op_cfg in node_qc_options_list]

    base_config = _create_node_single_candidate_qc(fw_info, weight_channel_axis, base_config, node_attrs_list)

    return candidates, base_config


def set_manual_bitwidth_config(graph, bit_width_config: BitWidthConfig):
    """
    Filters candidates per manual bit-width config.

    Args:
        graph: graph after candidates have been set on nodes.
        bit_width_config: bit-width config.
    """
    manual_activation_bitwidths = bit_width_config.get_nodes_to_manipulate_activation_bit_widths(graph)
    for n, a_nbits in manual_activation_bitwidths.items():
        candidates = [qc for qc in n.candidates_quantization_cfg if qc.activation_quantization_cfg.activation_n_bits == a_nbits]
        if not candidates:
            raise ValueError(f'Invalid manual bitwidth {a_nbits} for activation of node {n}. '
                             f'Only bitwidth supported by TPC can be specified.')
        n.candidates_quantization_cfg.activation_quantization_cfg = candidates
        n.base_quantization_cfg.activation_quantization_cfg.activation_n_bits = a_nbits

    manual_weights_bitwidths = bit_width_config.get_nodes_to_manipulate_weights_bit_widths(graph)

    def qc_attr_nbits(qc, attr, n):
        if attr not in qc.activation_quantization_cfg.weights_quantization_cfg.get_all_weights_attrs():
            raise ValueError(f'Invalid attribute {attr} in manual weights configuration for node {n}')
        return qc.activation_quantization_cfg.weights_quantization_cfg.get_attr_config(attr)

    for n, manual_wbits in manual_weights_bitwidths.items():
        candidates = [qc for qc in n.candidates_quantization_cfg
                      if all(qc_attr_nbits(qc, attr, n) == w_nbits for w_nbits, attr in manual_wbits)]
        if not candidates:
            raise ValueError(f'Invalid manual bitwidth configuration {manual_wbits} for node {n}. '
                             f'Only bitwidth supported by TPC can be specified.')
        n.candidates_quantization_cfg.weights_quantization_cfg = candidates
        for w_nbits, attr in manual_wbits:
            n.base_quantization_cfg.weights_quantization_cfg.get_attr_config(attr).weights_n_bits = w_nbits


