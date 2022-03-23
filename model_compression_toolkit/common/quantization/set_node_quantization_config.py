# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import List

from model_compression_toolkit.common import Logger, BaseNode
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.quantization.node_quantization_config import NodeActivationQuantizationConfig, \
    NodeWeightsQuantizationConfig
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn
from model_compression_toolkit.common.hardware_representation.hardware2framework import FrameworkHardwareModel
from model_compression_toolkit.common.hardware_representation.op_quantization_config import QuantizationConfigOptions, \
    OpQuantizationConfig


def set_quantization_configuration_to_graph(graph: Graph,
                                            quant_config: QuantizationConfig) -> Graph:
    """
    Add quantization configuration for each graph node.

    Args:
        graph: Graph for which to add quantization info to each node.
        quant_config: Quantization configuration containing parameters for how the graph should be quantized.

    Returns:
        The graph with quantization configurations attached to each node in it.
    """

    graph_with_qcs = copy.deepcopy(graph)
    for n in graph_with_qcs.nodes:
        set_quantization_configs_to_node(node=n,
                                         quant_config=quant_config,
                                         fw_info=graph.fw_info,
                                         fw_hw_model=graph.fw_hw_model)
    return graph_with_qcs


def set_quantization_configs_to_node(node: BaseNode,
                                     quant_config: QuantizationConfig,
                                     fw_info: FrameworkInfo,
                                     fw_hw_model: FrameworkHardwareModel):
    """
    Create and set quantization configurations to a node (for both weights and activation).

    Args:
        node: Node to set its quantization configurations.
        quant_config: Quantization configuration to generate the node's configurations from.
        fw_info: Information needed for quantization about the specific framework.
        fw_hw_model: FrameworkHardwareModel to get default OpQuantizationConfig.

    """
    op_cfg = fw_hw_model.get_default_op_qc()

    # Create activation QC for this node
    node.activation_quantization_cfg = create_node_activation_qc(quant_config,
                                                                 fw_info,
                                                                 op_cfg)

    enable_activation_quantization = quant_config.enable_activation_quantization and (fw_info.in_activation_ops(node) or fw_info.in_kernel_ops(node))
    node.activation_quantization_cfg.enable_activation_quantization = enable_activation_quantization

    # Create weights QC for this node
    weight_channel_axis = fw_info.kernel_channels_mapping.get(node.type)[0]
    node.candidates_weights_quantization_cfg = _create_node_candidates_weights_qc(quant_config,
                                                                                  fw_info,
                                                                                  weight_channel_axis,
                                                                                  op_cfg)

    enable_weights_quantization = quant_config.enable_weights_quantization and fw_info.in_kernel_ops(node)
    for qc in node.candidates_weights_quantization_cfg:
        qc.enable_weights_quantization = enable_weights_quantization


def create_node_activation_qc(qc: QuantizationConfig,
                              fw_info: FrameworkInfo,
                              op_cfg: OpQuantizationConfig) -> NodeActivationQuantizationConfig:
    """
    Create a activations quantization configuration from a QuantizationConfig object.

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
        Logger.critical('Unknown quantization method for activations')

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method,
                                                                              qc.activation_error_method)

    return NodeActivationQuantizationConfig(qc,
                                            op_cfg,
                                            activation_quantization_fn,
                                            activation_quantization_params_fn)


def create_node_weights_qc(qc: QuantizationConfig,
                           fw_info: FrameworkInfo,
                           weight_channel_axis: int,
                           op_cfg: OpQuantizationConfig) -> NodeWeightsQuantizationConfig:
    """
    Create a weights quantization configuration from a QuantizationConfig object.

    Args:
        op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its
        weights/activations should be quantized)
        weight_channel_axis: Axis to quantize a node's kernel when quantizing per-channel.

    Returns:
        Weights quantization configuration of a node.
    """

    weights_quantization_fn = fw_info.weights_quantizer_mapping.get(op_cfg.weights_quantization_method)

    if weights_quantization_fn is None:
        Logger.critical('Unknown quantization method for weights')

    weights_quantization_params_fn = get_weights_quantization_params_fn(op_cfg.weights_quantization_method,
                                                                        qc.weights_error_method)

    return NodeWeightsQuantizationConfig(qc,
                                         op_cfg,
                                         weights_quantization_fn,
                                         weights_quantization_params_fn,
                                         weight_channel_axis)


def _create_node_candidates_weights_qc(qc: QuantizationConfig,
                                       fw_info: FrameworkInfo,
                                       weight_channel_axis: int,
                                       op_cfg: OpQuantizationConfig) -> List[NodeWeightsQuantizationConfig]:
    """
    Create a list of candidates of weights quantization configurations for a node.

    Args:
        qc: Quantization configuration the quantization process should follow.
        fw_info: Framework information (e.g., which layers should have their kernels' quantized).
        weight_channel_axis: Output channel index of the node's kernel.
        op_cfg: OpQuantizationConfig for the node with quantizers types.

    Returns:
        List of candidates of weights quantization configurations to set for a node.
    """

    candidats = []
    if isinstance(qc, MixedPrecisionQuantizationConfig):
        qc.weights_n_bits.sort(reverse=True)
        for nbits in qc.weights_n_bits:
            single_nbits_qc = copy.deepcopy(qc)
            single_nbits_qc.weights_n_bits = nbits
            candidats.append(create_node_weights_qc(single_nbits_qc,
                                                    fw_info,
                                                    weight_channel_axis,
                                                    op_cfg))
    else:
        candidats.append(create_node_weights_qc(qc,
                                                fw_info,
                                                weight_channel_axis,
                                                op_cfg))

    return candidats
