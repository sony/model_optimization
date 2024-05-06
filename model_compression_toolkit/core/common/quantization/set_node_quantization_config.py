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
from typing import List, Tuple

from model_compression_toolkit.core.common import BaseNode
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
                                            mixed_precision_enable: bool = False,
                                            running_gptq: bool = False) -> Graph:
    """
    Add quantization configuration for each graph node.

    Args:
        graph: Graph for which to add quantization info to each node.
        quant_config: Quantization configuration containing parameters for how the graph should be quantized.
        mixed_precision_enable: is mixed precision enabled.
        running_gptq: Whether or not a GPTQ optimization is planned to run after the PTQ process.

    Returns:
        The graph with quantization configurations attached to each node in it.
    """

    if quant_config.weights_error_method == QuantizationErrorMethod.HMSE:
        if not running_gptq:
            Logger.warning(f"The HMSE error method for parameters selection is only supported when running GPTQ "
                           f"optimization due to long execution time that is not suitable for basic PTQ. "
                           f"Using the default MSE error method instead.")
            quant_config.weights_error_method = QuantizationErrorMethod.MSE
        else:
            Logger.warning("Using the HMSE error method for weights quantization parameters search. "
                           "Note: This method may significantly increase runtime during the parameter search process.")

    for n in graph.nodes:
        set_quantization_configs_to_node(node=n,
                                         quant_config=quant_config,
                                         fw_info=graph.fw_info,
                                         tpc=graph.tpc,
                                         mixed_precision_enable=mixed_precision_enable)
    return graph


def set_quantization_configs_to_node(node: BaseNode,
                                     quant_config: QuantizationConfig,
                                     fw_info: FrameworkInfo,
                                     tpc: TargetPlatformCapabilities,
                                     mixed_precision_enable: bool = False):
    """
    Create and set quantization configurations to a node (for both weights and activation).

    Args:
        node: Node to set its quantization configurations.
        quant_config: Quantization configuration to generate the node's configurations from.
        fw_info: Information needed for quantization about the specific framework.
        tpc: TargetPlatformCapabilities to get default OpQuantizationConfig.
        mixed_precision_enable: is mixed precision enabled.
    """
    node_qc_options = node.get_qco(tpc)

    # Create QC candidates for weights and activation combined
    weight_channel_axis = fw_info.kernel_channels_mapping.get(node.type)
    node.candidates_quantization_cfg = _create_node_candidates_qc(quant_config,
                                                                  fw_info,
                                                                  weight_channel_axis,
                                                                  node_qc_options,
                                                                  node,
                                                                  mixed_precision_enable=mixed_precision_enable)

    # sorting the candidates by kernel attribute weights number of bits first and then by activation number of bits
    # (in reversed order). since only kernel attribute is quantized in weights mixed precision,
    # if the node doesn't have a kernel attribute, we only sort by activation_n_bits.
    node.sort_node_candidates(fw_info)

    for candidate_qc in node.candidates_quantization_cfg:
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
                               node_qc_options: QuantizationConfigOptions,
                               node: BaseNode,
                               mixed_precision_enable: bool = False) -> List[CandidateNodeQuantizationConfig]:
    """
    Create a list of candidates of weights and activation quantization configurations for a node.

    Args:
        qc: Quantization configuration the quantization process should follow.
        fw_info: Framework information (e.g., which layers should have their kernels' quantized).
        weight_channel_axis: (Output, Input) channel index of the node's kernel.
        node_qc_options: QuantizationConfigOptions for the node with quantization candidates information.
        node: A node to set quantization configuration candidates to.
        mixed_precision_enable: is mixed precision enabled

    Returns:
        List of candidates of weights quantization configurations to set for a node.
    """

    candidates = []
    node_attrs_list = node.get_node_weights_attributes()

    if mixed_precision_enable:
        for op_cfg in node_qc_options.quantization_config_list:
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
                                                           node_qc_options.base_config,
                                                           node_attrs_list))

    return candidates
