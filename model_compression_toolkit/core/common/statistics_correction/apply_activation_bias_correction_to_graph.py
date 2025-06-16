# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core import CoreConfig, QuantizationConfig
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.quantization.node_quantization_config import WeightsAttrQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import AttributeQuantizationConfig


def apply_activation_bias_correction_to_graph(graph: Graph,
                                              core_config: CoreConfig,
                                              fw_impl: FrameworkImplementation) -> Graph:
    """
    Get a graph, where each node has a final activation quantization configuration (with an activation bias
    correction term in it), and apply the activation bias correction for each node in the graph.

    Args:
        graph: Graph to apply activation bias correction to.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with activation bias correction apply to it's nodes.
    """

    for n in graph.nodes:
        # Activation bias correction is only relevant for nodes with kernel op
        if core_config.quantization_config.activation_bias_correction and n.kernel_attr is not None and \
                n.final_activation_quantization_cfg.activation_bias_correction_term is not None:
            # If activation bias correction is enabled in n.quantization_cfg, an activation bias correction term was
            # calculated during model preparation, and is used now in the node's bias term.
            _apply_activation_bias_correction_to_node(n, fw_impl, core_config.quantization_config)
    return graph


def _apply_activation_bias_correction_to_node(node: BaseNode,
                                              fw_impl: FrameworkImplementation,
                                              qc: QuantizationConfig):
    """
    Set new bias to node using the activation bias correction term that is stored in the
    final activation quantization configuration.

    Args:
        node: Node to set its corrected bias after activation bias correction.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        qc: QuantizationConfig containing parameters of how the model should be quantized.

    """
    correction = node.final_activation_quantization_cfg.activation_bias_correction_term
    bias = node.get_weights_by_keys(fw_impl.constants.BIAS)  # get original bias from node's weights

    if bias is None:
        # If the layer has no bias, we set the bias as -correction.
        node.set_weights_by_keys(fw_impl.constants.BIAS, - correction)

        # Mark the use_bias attribute of the node.
        node.framework_attr[fw_impl.constants.USE_BIAS] = True

        # Configure the quantization of the bias as disabled.
        node.final_weights_quantization_cfg.set_attr_config(fw_impl.constants.BIAS,
                                                            WeightsAttrQuantizationConfig(
                                                                qc,
                                                                AttributeQuantizationConfig(
                                                                    enable_weights_quantization=False)))
    else:
        # If the layer has bias, we subtract the correction from original bias
        node.set_weights_by_keys(fw_impl.constants.BIAS, bias - correction)
