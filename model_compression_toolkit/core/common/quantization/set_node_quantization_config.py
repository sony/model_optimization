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
from typing import List, Tuple, Dict

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.constants import POSITIONAL_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import max_input_activation_n_bits
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


# TODO irena refactor (if needed) and move to load_fqc
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
    from model_compression_toolkit.quantization_preparation.load_fqc import fetch_qc_options_for_node

    # Filter quantization config options that don't match the graph.
    _base_config = node_qc_options.base_config
    _node_qc_options = node_qc_options.quantization_configurations

    # Build next_nodes list by appending to the node's next nodes list all nodes that are quantization preserving.
    _next_nodes = graph.get_next_nodes(node)
    next_nodes = []
    while len(_next_nodes):
        n = _next_nodes.pop(0)
        qco = fetch_qc_options_for_node(n, fqc)
        qp = [qc.quantization_preserving for qc in qco.quantization_configurations]
        if not all(qp) and any(qp):
            Logger.error(f'Attribute "quantization_preserving" should be the same for all QuantizaionConfigOptions in {n}.')
        if qp[0]:
            _next_nodes.extend(graph.get_next_nodes(n))
        next_nodes.append(n)

    if len(next_nodes) == 0:
        return _base_config, _node_qc_options

    next_nodes_qc_options = [fetch_qc_options_for_node(_node, fqc) for _node in next_nodes]
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


def set_manual_bitwidth_config(graph, bit_width_config: BitWidthConfig):
    """
    Filters candidates per manual bit-width config.

    Args:
        graph: graph after candidates have been set on nodes.
        bit_width_config: bit-width config.
    """
    manual_activation_bitwidths = bit_width_config.get_nodes_activation_bit_widths(graph)
    manual_weights_bitwidths = bit_width_config.get_nodes_weights_bit_widths(graph)

    if manual_activation_bitwidths:
        _set_manual_activation_bitwidths(manual_activation_bitwidths)

    if manual_weights_bitwidths:
        _set_manual_weights_bitwidths(manual_weights_bitwidths)


# TODO irena: check coverage and add missing tests
def _set_manual_activation_bitwidths(manual_activation_bitwidths: Dict[BaseNode, int]):
    """
    Filters out candidates that don't match the requested manual activation bitwidths, and updates the
    activation bitwidth in the base quantization config.

    Args:
        manual_activation_bitwidths: nodes' manual activation bitwidth.

    Raises:
        ValueError: if the manual bitwidth is requested for un-quantized node.
                    if the manual bitwidth is not compatible with any candidate.
    """
    for n, a_nbits in manual_activation_bitwidths.items():
        quant_mode = n.quantization_cfg.get_activation_quant_mode()
        # TODO irena: for FLN I think it should be ignored with warning for layer filter, and error for name filter
        if quant_mode != ActivationQuantizationMode.QUANT:
            raise ValueError(f'Cannot apply manual activation bit-width for node {n} with activation quantization mode'
                             f'{quant_mode}, as it does not have its own quantization configuration.')
        candidates = [qc for qc in n.candidates_quantization_cfg
                      if qc.activation_quantization_cfg.activation_n_bits == a_nbits]
        if not candidates:
            valid_nbits = sorted(list({qc.activation_quantization_cfg.activation_n_bits
                                       for qc in n.candidates_quantization_cfg}))
            raise ValueError(
                f'Manually selected activation bit-width {a_nbits} is invalid for node {n}. '
                f'Valid bit-widths: {valid_nbits}.')
        n.quantization_cfg.candidates_quantization_cfg = candidates
        n.quantization_cfg.base_quantization_cfg.activation_quantization_cfg.activation_n_bits = a_nbits


# TODO irena: check coverage
def _set_manual_weights_bitwidths(manual_weights_bitwidths: Dict[BaseNode, Dict[str, int]]):
    """
    Filters out candidates that don't match the requested weight attributes manual bitwidths, and updates the bitwidths
    in the base quantization config.

    Args:
        manual_activation_bitwidths: nodes' manual activation bitwidth.

    Raises:
        ValueError: if the manual bitwidth is requested for non-existing attribute.
                    if the manual bitwidth is requested for un-quantized weights attribute.
                    if the manual bitwidth is not compatible with any candidate.
    """
    def qc_attr_nbits(qc, attr, n):
        if attr == POSITIONAL_ATTR:
            pos_attrs = qc.weights_quantization_cfg.pos_attributes_config_mapping
            if not pos_attrs:
                raise ValueError('Unexpected positional attribute in manual weights bit-width for node {n}.')
            if any(cfg.enable_weights_quantization is False for cfg in pos_attrs.values()):
                raise ValueError(f'Cannot apply manual bit-width configuration for positional attribute of node {n} as '
                                 f'the attribute is not quantized.')
            assert len({cfg.weights_n_bits for cfg in pos_attrs.values()}) == 1
            return list(pos_attrs.values())[0].weights_n_bits
        if attr not in qc.weights_quantization_cfg.all_weight_attrs:
            raise ValueError(f'Unexpected attribute {attr} in manual weights bit-width configuration for node {n}.')
        attr_cfg = qc.weights_quantization_cfg.get_attr_config(attr)
        if not attr_cfg.enable_weights_quantization:
            raise ValueError(f'Cannot apply manual bit-width configuration for weights attribute {attr} of node {n} as '
                             f'the attribute is not quantized.')
        return qc.weights_quantization_cfg.get_attr_config(attr).weights_n_bits

    for n, manual_wbits in manual_weights_bitwidths.items():
        candidates = [qc for qc in n.candidates_quantization_cfg
                      if all(qc_attr_nbits(qc, attr, n) == w_nbits for attr, w_nbits in manual_wbits.items())]
        if not candidates:
            raise ValueError(f'Cannot apply manual weights bit-width configuration {manual_wbits} for node {n} as it '
                             f'does not match any of the quantization candidates.')
        n.quantization_cfg.candidates_quantization_cfg = candidates
        for attr, w_nbits in manual_wbits.items():
            base_weights_cfg = n.quantization_cfg.base_quantization_cfg.weights_quantization_cfg
            if attr == POSITIONAL_ATTR:
                for pos_attr in base_weights_cfg.pos_attributes_config_mapping:
                    base_weights_cfg.get_attr_config(pos_attr).weights_n_bits = w_nbits
            else:
                base_weights_cfg.get_attr_config(attr).weights_n_bits = w_nbits
