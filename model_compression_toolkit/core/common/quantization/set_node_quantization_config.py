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


import copy
from typing import List, Tuple,  Optional

from mct_quantizers.common.constants import ACTIVATION_N_BITS
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig, \
    QuantizationErrorMethod
from model_compression_toolkit.core.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn
from model_compression_toolkit.core.common.quantization.quantization_fn_selection import \
    get_weights_quantization_fn
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.target_platform.op_quantization_config import OpQuantizationConfig, \
    QuantizationConfigOptions


def set_quantization_configuration_to_graph(graph: Graph,
                                            quant_config: QuantizationConfig,
                                            bit_width_config: BitWidthConfig = None,
                                            mixed_precision_enable: bool = False,
                                            running_gptq: bool = False) -> Graph:
    """
    Add quantization configuration for each graph node.

    Args:
        graph (Graph): Graph for which to add quantization info to each node.
        quant_config (QuantizationConfig): Quantization configuration containing parameters for how the graph should be quantized.
        bit_width_config (BitWidthConfig): Configuration for manual bit width selection. Defaults to None.
        mixed_precision_enable (bool): Whether mixed precision is enabled. Defaults to False.
        running_gptq (bool): Whether or not a GPTQ optimization is planned to run after the PTQ process. Defaults to False.

    Returns:
        Graph: The graph with quantization configurations attached to each node in it.
    """

    if quant_config.weights_error_method == QuantizationErrorMethod.HMSE:
        if not running_gptq:
            raise ValueError(f"The HMSE error method for parameters selection is only supported when running GPTQ "
                             f"optimization due to long execution time that is not suitable for basic PTQ.")
        Logger.warning("Using the HMSE error method for weights quantization parameters search. "
                       "Note: This method may significantly increase runtime during the parameter search process.")

    nodes_to_manipulate_bit_widths = {} if bit_width_config is None else bit_width_config.get_nodes_to_manipulate_bit_widths(graph)

    for n in graph.nodes:
        set_quantization_configs_to_node(node=n,
                                         graph=graph,
                                         quant_config=quant_config,
                                         fw_info=graph.fw_info,
                                         tpc=graph.tpc,
                                         mixed_precision_enable=mixed_precision_enable,
                                         manual_bit_width_override=nodes_to_manipulate_bit_widths.get(n))
    return graph


def filter_node_qco_by_graph(node: BaseNode,
                             tpc: TargetPlatformCapabilities,
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
        tpc: TPC to extract the QuantizationConfigOptions for the next nodes.
        graph: Graph object.
        node_qc_options: Node's QuantizationConfigOptions.

    Returns:
        A base config (OpQuantizationConfig) and a config options list (list of OpQuantizationConfig)
        that are compatible with next nodes supported input bit-widths.

    """
    # Filter quantization config options that don't match the graph.
    _base_config = node_qc_options.base_config
    _node_qc_options = node_qc_options.quantization_config_list

    # Build next_nodes list by appending to the node's next nodes list all nodes that are quantization preserving.
    _next_nodes = graph.get_next_nodes(node)
    next_nodes = []
    while len(_next_nodes):
        n = _next_nodes.pop(0)
        qco = n.get_qco(tpc)
        qp = [qc.quantization_preserving for qc in qco.quantization_config_list]
        if not all(qp) and any(qp):
            Logger.error(f'Attribute "quantization_preserving" should be the same for all QuantizaionConfigOptions in {n}.')
        if qp[0]:
            _next_nodes.extend(graph.get_next_nodes(n))
        next_nodes.append(n)

    if len(next_nodes):
        next_nodes_qc_options = [_node.get_qco(tpc) for _node in next_nodes]
        next_nodes_supported_input_bitwidth = min([op_cfg.max_input_activation_n_bits
                                                   for qc_opts in next_nodes_qc_options
                                                   for op_cfg in qc_opts.quantization_config_list])

        # Filter node's QC options that match next nodes input bit-width.
        _node_qc_options = [_option for _option in _node_qc_options
                            if _option.activation_n_bits <= next_nodes_supported_input_bitwidth]
        if len(_node_qc_options) == 0:
            Logger.critical(f"Graph doesn't match TPC bit configurations: {node} -> {next_nodes}.")

        # Verify base config match
        if any([node_qc_options.base_config.activation_n_bits > qc_opt.base_config.max_input_activation_n_bits
                for qc_opt in next_nodes_qc_options]):
            # base_config activation bits doesn't match next node supported input bit-width -> replace with
            # a qco from quantization_config_list with maximum activation bit-width.
            if len(_node_qc_options) > 0:
                output_act_bitwidth = {qco.activation_n_bits: i for i, qco in enumerate(_node_qc_options)}
                _base_config = _node_qc_options[output_act_bitwidth[max(output_act_bitwidth)]]
                Logger.warning(f"Node {node} base quantization config changed to match Graph and TPC configuration.\nCause: {node} -> {next_nodes}.")
            else:
                Logger.critical(f"Graph doesn't match TPC bit configurations: {node} -> {next_nodes}.")  # pragma: no cover

    return _base_config, _node_qc_options


def set_quantization_configs_to_node(node: BaseNode,
                                     graph: Graph,
                                     quant_config: QuantizationConfig,
                                     fw_info: FrameworkInfo,
                                     tpc: TargetPlatformCapabilities,
                                     mixed_precision_enable: bool = False,
                                     manual_bit_width_override: Optional[int] = None):
    """
    Create and set quantization configurations to a node (for both weights and activation).

    Args:
        node (BaseNode): Node to set its quantization configurations.
        graph (Graph): Model's internal representation graph.
        quant_config (QuantizationConfig): Quantization configuration to generate the node's configurations from.
        fw_info (FrameworkInfo): Information needed for quantization about the specific framework.
        tpc (TargetPlatformCapabilities): TargetPlatformCapabilities to get default OpQuantizationConfig.
        mixed_precision_enable (bool): Whether mixed precision is enabled. Defaults to False.
        manual_bit_width_override (Optional[int]): Specifies a custom bit-width to override the node's activation bit-width. Defaults to None.
    """
    node_qc_options = node.get_qco(tpc)
    base_config, node_qc_options_list = filter_node_qco_by_graph(node, tpc, graph, node_qc_options)

    # If a manual_bit_width_override is given, filter node_qc_options_list to retain only the options with activation bits equal to manual_bit_width_override,
    # and update base_config accordingly.
    base_config, node_qc_options_list = filter_qc_options_with_manual_bit_width(
        node=node,
        node_qc_options_list=node_qc_options_list,
        base_config=base_config,
        manual_bit_width_override=manual_bit_width_override,
        mixed_precision_enable=mixed_precision_enable)

    # Create QC candidates for weights and activation combined
    weight_channel_axis = fw_info.kernel_channels_mapping.get(node.type)
    node.candidates_quantization_cfg = _create_node_candidates_qc(quant_config,
                                                                  fw_info,
                                                                  weight_channel_axis,
                                                                  node_qc_options_list,
                                                                  base_config,
                                                                  node,
                                                                  mixed_precision_enable=mixed_precision_enable)

    # sorting the candidates by kernel attribute weights number of bits first and then by activation number of bits
    # (in reversed order). since only kernel attribute is quantized in weights mixed precision,
    # if the node doesn't have a kernel attribute, we only sort by activation_n_bits.
    node.sort_node_candidates(fw_info)

    for candidate_qc in node.candidates_quantization_cfg:
        if node.name == 'input_ids':
            candidate_qc.activation_quantization_cfg.enable_activation_quantization = False
        else:
            candidate_qc.activation_quantization_cfg.enable_activation_quantization = \
                candidate_qc.activation_quantization_cfg.enable_activation_quantization and node.get_has_activation()


def create_node_activation_qc(qc: QuantizationConfig,
                              fw_info: FrameworkInfo,
                              op_cfg: OpQuantizationConfig) -> NodeActivationQuantizationConfig:
    """
    Create an activation quantization configuration from a QuantizationConfig object.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its
        weights/activations should be quantized)
        op_cfg: OpQuantizationConfig with quantizers types to set in node quantization configuration.

    Returns:
        Activation quantization configuration of a node.
    """

    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(op_cfg.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown activation quantization method specified.')  # pragma: no cover

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method)

    return NodeActivationQuantizationConfig(qc,
                                            op_cfg,
                                            activation_quantization_fn,
                                            activation_quantization_params_fn)


def _create_node_single_candidate_qc(qc: QuantizationConfig,
                                     fw_info: FrameworkInfo,
                                     weight_channel_axis: Tuple[int, int],
                                     op_cfg: OpQuantizationConfig,
                                     node_attrs_list: List[str]) -> CandidateNodeQuantizationConfig:
    """
    Create quantization configuration candidate from a QuantizationConfig object.
    Creates both weights and activation quantization configurations
    and initialize a candidate object that encapsulates both.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether its
            weights/activations should be quantized)
        weight_channel_axis: (Output, Input) channel index of the node's kernel.
        op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
        node_attrs_list: A list of the node's weights attributes names.

    Returns: a CandidateNodeQuantizationConfig object with both weights and activation quantization config objects.

    """

    # parameters for weights attributes quantization are set within  CandidateNodeQuantizationConfig initialization

    # get parameters for activation quantization
    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(op_cfg.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown activation quantization method specified.')  # pragma: no cover

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method)

    # TODO: remove this validation and warning once enabling all attributes quantization by default
    attrs_with_enabled_quantization = [attr for attr, cfg in op_cfg.attr_weights_configs_mapping.items()
                                       if cfg.enable_weights_quantization]
    if len(attrs_with_enabled_quantization) > 1:
        Logger.warning(f"Multiple weights attributes quantization is enabled via the provided TPC."
                       f"Quantizing any attribute other than the kernel is experimental "
                       f"and may be subject to unstable behavior."
                       f"Attributes with enabled weights quantization: {attrs_with_enabled_quantization}.")

    return CandidateNodeQuantizationConfig(qc=qc,
                                           op_cfg=op_cfg,
                                           activation_quantization_fn=activation_quantization_fn,
                                           activation_quantization_params_fn=activation_quantization_params_fn,
                                           weights_channels_axis=weight_channel_axis,
                                           node_attrs_list=node_attrs_list)


def _create_node_candidates_qc(qc: QuantizationConfig,
                               fw_info: FrameworkInfo,
                               weight_channel_axis: Tuple[int, int],
                               node_qc_options_list: List[OpQuantizationConfig],
                               base_config: OpQuantizationConfig,
                               node: BaseNode,
                               mixed_precision_enable: bool = False) -> List[CandidateNodeQuantizationConfig]:
    """
    Create a list of candidates of weights and activation quantization configurations for a node.

    Args:
        qc (QuantizationConfig): Quantization configuration the quantization process should follow.
        fw_info (FrameworkInfo): Framework information (e.g., which layers should have their kernels quantized).
        weight_channel_axis (Tuple[int, int]): (Output, Input) channel index of the node's kernel.
        node_qc_options_list (List[OpQuantizationConfig]): List of quantization configs of node.
        base_config (OpQuantizationConfig): Base quantization config for node.
        node (BaseNode): A node to set quantization configuration candidates to.
        mixed_precision_enable (bool): Whether mixed precision is enabled. Defaults to False.

    Returns:
        List[CandidateNodeQuantizationConfig]: List of candidates of weights quantization configurations to set for a node.
    """

    candidates = []
    node_attrs_list = node.get_node_weights_attributes()

    if mixed_precision_enable:
        for op_cfg in node_qc_options_list:
            candidate_qc = copy.deepcopy(qc)
            candidates.append(_create_node_single_candidate_qc(candidate_qc,
                                                               fw_info,
                                                               weight_channel_axis,
                                                               op_cfg,
                                                               node_attrs_list))

    else:
        candidates.append(_create_node_single_candidate_qc(qc,
                                                           fw_info,
                                                           weight_channel_axis,
                                                           base_config,
                                                           node_attrs_list))

    return candidates


def filter_qc_options_with_manual_bit_width(
        node: BaseNode,
        node_qc_options_list: List[OpQuantizationConfig],
        base_config: OpQuantizationConfig,
        manual_bit_width_override: Optional[int],
        mixed_precision_enable: bool) -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
    """
    Update the quantization configurations for a node, allowing manual bit-width overrides if specified.

    Args:
        node (BaseNode): A node to set quantization configuration candidates to.
        node_qc_options_list (List[OpQuantizationConfig]): List of quantization configs for the node.
        base_config (OpQuantizationConfig): Base quantization config for the node.
        manual_bit_width_override (Optional[int]): Specifies a custom bit-width to override the node's activation bit-width.
        mixed_precision_enable (bool): Whether mixed precision is enabled.

    Returns:
        Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]: The updated base configuration and the filtered list of quantization configs.
    """
    if manual_bit_width_override is None:
        return base_config, node_qc_options_list

    # Filter node_qc_options_list to retain only the options with activation bits equal to manual_bit_width_override.
    node_qc_options_list = [op_cfg for op_cfg in node_qc_options_list if
                                manual_bit_width_override == op_cfg.activation_n_bits]

    if len(node_qc_options_list) == 0:
        Logger.critical(f"Manually selected activation bit-width {manual_bit_width_override} is invalid for node {node}.")
    else:
        # Update the base_config to one of the values from the filtered node_qc_options_list.
        # First, check if a configuration similar to the original base_config but with activation bits equal to manual_bit_width_override exists.
        # If it does, use it as the base_config. If not, choose a different configuration from node_qc_options_list.
        Logger.info(f"Setting node {node} bit-width to manually selected bit-width: {manual_bit_width_override} bits.")
        updated_base_config = base_config.clone_and_edit({ACTIVATION_N_BITS, manual_bit_width_override})
        if updated_base_config in node_qc_options_list:
            # If a base_config with the specified manual_bit_width_override exists in the node_qc_options_list,
            # point the base_config to this option.
            base_config = node_qc_options_list[node_qc_options_list.index(updated_base_config)]
        else:
            # Choose a different configuration from node_qc_options_list. If multiple options exist, issue a warning.
            base_config = node_qc_options_list[0]
            if len(node_qc_options_list) > 0 and not mixed_precision_enable:
                Logger.info(
                    f"Request received to select {manual_bit_width_override} activation bits. However, the base configuration for layer type {node.type} is missing in the node_qc_options_list."
                    f" Overriding base_config with an option that uses {manual_bit_width_override} bit activations.")  # pragma: no cover
    return base_config, node_qc_options_list