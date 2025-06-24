# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Optional

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_info import ChannelAxisMapping
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig, NodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import filter_node_qco_by_graph
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities, \
    QuantizationConfigOptions, OpQuantizationConfig


def load_fqc_configuration(graph: Graph, fqc: FrameworkQuantizationCapabilities):
    """
    Set-up graph for quantization per TPC.
    Each node will contain quantization candidates for mixed precision and the base config for single precision.
    The graph will contain the fusing info.

    Args:
        graph: graph.
        fqc: framework quantization capabilities object.

    Returns:
        Updated graph.
    """
    graph = _set_nodes_quantization_configuration(graph, fqc)
    graph = _set_fusion_info(graph, fqc)

    return graph


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
    qc_options = fetch_qc_options_for_node(node, fqc)
    base_config, candidates_qcs = filter_node_qco_by_graph(node, fqc, graph, qc_options)

    node_attrs_list = node.get_node_weights_attributes()
    mp_candidates = [_create_candidate(node.channel_axis, qc, node_attrs_list)
                     for qc in candidates_qcs]
    sp_cfg = _create_candidate(node.channel_axis, base_config, node_attrs_list)

    node.quantization_cfg = NodeQuantizationConfig(base_quantization_cfg=sp_cfg,
                                                   candidates_quantization_cfg=mp_candidates)

    # TODO is not needed anymore as find min/max candidate look for a real max/min, but some tests still count on it
    node.sort_node_candidates()

    if not node.has_activation:
        node.quantization_cfg.update_activation_quantization_mode(ActivationQuantizationMode.NO_QUANT)

    _disable_unsupported_quant_preserving(node, graph)


def fetch_qc_options_for_node(node: BaseNode,
                              fqc: FrameworkQuantizationCapabilities,
                              return_default=True) -> Optional[QuantizationConfigOptions]:
    """
    Get quantization configuration options for the node from TPC.

    Args:
        node: node for which to fetch quantization configuration.
        fqc: framework quantization capabilities.
        return_default: whether to return the default qco or None if node op is not in FQC.

    Returns:
        Quantization configuration options for the node.
    """
    # qcos by filters
    filter_matches = [(fl, qco) for fl, qco in fqc.filterlayer2qco.items() if node.is_match_filter_params(fl)]
    fls, filter_qcos = zip(*filter_matches) if filter_matches else (None, None)
    if filter_qcos and any(qco != filter_qcos[0] for qco in filter_qcos[1:]):
        raise ValueError(f'Cannot assign quantization configuration to {node} as it matches more than one filter with '
                         f'conflicting configs: {fls}.')

    # qco by opset
    # must use is_match_type for functional op in TF2.15
    matches = [(op_type, qco) for op_type, qco in fqc.layer2qco.items() if node.is_match_type(op_type)]
    op_types, qcos = zip(*matches) if matches else (None, None)
    if qcos and any(qco != qcos[0] for qco in qcos[1:]):
        raise ValueError(f'Cannot assign quantization configuration to {node} as it matches more than one op type with '
                         f'conflicting configs: {op_types}.')

    # if node matches by both filter and opset, filter takes priority
    if filter_qcos:
        return filter_qcos[0]

    if qcos:
        return qcos[0]

    return fqc.tpc.default_qco if return_default else None


def _set_nodes_quantization_configuration(graph: Graph,
                                          fqc: FrameworkQuantizationCapabilities) -> Graph:
    """
    Set quantization configuration for each graph node.

    Args:
        graph: graph to set with quantization configuration.
        fqc: framework quantization capabilities.

    Returns:
        Graph: The graph with quantization configurations attached to each node in it.
    """
    _validate_custom_ops_have_qco(graph, fqc)

    for n in graph.get_topo_sorted_nodes():
        set_quantization_configs_to_node(node=n,
                                         graph=graph,
                                         fqc=fqc)
    return graph


def _set_fusion_info(graph: Graph, fqc: FrameworkQuantizationCapabilities) -> Graph:
    """

    Args:
        graph: graph.
        fqc: quantization capabilities with attached framework.

    Returns:

    """
    # TODO fix the dict with const keys inside get_fusing_patterns. use named tuple or class
    # TODO irena instead of storing fusion inside graph (including tpc objects) and then let graph convert tpc op config to
    #  node config, do it here and only store in graph whatever is relevant after this stage.
    fusing_info = FusingInfoGenerator(fqc.get_fusing_patterns()).generate_fusing_info(graph)
    graph.fusing_info = fusing_info
    graph.override_fused_node_activation_quantization_candidates()
    return graph


def _disable_unsupported_quant_preserving(node: BaseNode, graph: Graph):
    """
    Disable quantization for quantization preserving ops in cases it cannot be supported
    (multiple inputs or un-quantized previous node).

    Args:
        node: current node.
        graph: graph.
    """
    if not node.quantization_cfg.get_activation_quant_mode() == ActivationQuantizationMode.PRESERVE_QUANT:
        return

    prev_nodes = graph.get_prev_nodes(node)
    if len(prev_nodes) != 1:
        Logger.info(f'Disabling Quantization-Preserving for node {node.name} with {len(prev_nodes)} inputs.')
        node.quantization_cfg.update_activation_quantization_mode(ActivationQuantizationMode.NO_QUANT)
    elif prev_nodes[0].quantization_cfg.get_activation_quant_mode() == ActivationQuantizationMode.NO_QUANT:
        Logger.info(f'Disabling Quantization-Preserving for node {node.name} since previous node activation '
                    f'quantization is disabled.')
        node.quantization_cfg.update_activation_quantization_mode(ActivationQuantizationMode.NO_QUANT)


# TODO irena copied from graph.set_fqc as is. Why does it have Keras errors?
def _validate_custom_ops_have_qco(graph, fqc):
    custom_nodes = [n for n in graph.nodes if n.is_custom]
    for n in custom_nodes:
        qco = fetch_qc_options_for_node(n, fqc, return_default=False)
        if not qco:
            Logger.critical(f'MCT does not support optimizing Keras custom layers. Found a layer of type {n.type}. '
                            ' Please add the custom layer to Framework Quantization Capabilities (FQC), or file a feature '
                            'request or an issue if you believe this should be supported.')  # pragma: no cover
        if any([qc.default_weight_attr_config.enable_weights_quantization for qc in qco.quantization_configurations]):
            Logger.critical(f'Layer identified: {n.type}. MCT does not support weight quantization for Keras custom layers.')  # pragma: no cover


def _create_candidate(weight_channel_axis: ChannelAxisMapping,
                      op_cfg: OpQuantizationConfig,
                      node_attrs_list: List[str]) -> CandidateNodeQuantizationConfig:
    """
    Create quantization configuration candidate.

    Args:
        weight_channel_axis: channels axes of the node's kernel.
        op_cfg: quantization config for the op.
        node_attrs_list: A list of the node's weights attributes names.

    Returns:
        Candidate quantization config.
    """

    aqc = NodeActivationQuantizationConfig(op_cfg=op_cfg)

    # TODO: remove this validation and warning once enabling all attributes quantization by default
    attrs_with_enabled_quantization = [attr for attr, cfg in op_cfg.attr_weights_configs_mapping.items()
                                       if cfg.enable_weights_quantization]
    if len(attrs_with_enabled_quantization) > 1:
        Logger.warning(f"Multiple weights attributes quantization is enabled via the provided FQC."
                       f"Quantizing any attribute other than the kernel is experimental "
                       f"and may be subject to unstable behavior."
                       f"Attributes with enabled weights quantization: {attrs_with_enabled_quantization}.")
    wqc = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                        weights_channels_axis=weight_channel_axis,
                                        node_attrs_list=node_attrs_list)

    return CandidateNodeQuantizationConfig(activation_quantization_cfg=aqc, weights_quantization_cfg=wqc)
