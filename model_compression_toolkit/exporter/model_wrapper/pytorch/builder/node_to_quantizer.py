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

from typing import Dict, Any

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX, \
    SCALE_PER_CHANNEL, LUT_VALUES
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig
from model_compression_toolkit.logger import Logger
from mct_quantizers import QuantizationMethod
from mct_quantizers import QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers import \
    constants as qi_inferable_quantizers_constants
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer
import numpy as np


def get_weights_inferable_quantizer_kwargs(node_qc: NodeWeightsQuantizationConfig, attr_name: str) -> Dict[str, Any]:
    """
    Get the quantization parameters for a weights inferable quantizer.
    Args:
        node_qc: The node quantization configuration of the node for which the quantizer is being created.
            Needs to match the specific quantization target.
        attr_name: The weights attribute to get its quantizer kwargs (if target is weights quantization).


    Returns:
        The quantization parameters as a dictionary.
    """

    if not isinstance(node_qc, NodeWeightsQuantizationConfig):
        Logger.critical(
            f"Non-compatible node quantization config was given for quantization target Weights.")  # pragma: no cover

    if attr_name is None:
        Logger.error(f"Attribute name was not specified for retrieving weights quantizer kwargs.")

    attr_node_qc = node_qc.get_attr_config(attr_name=attr_name)

    quantization_method = attr_node_qc.weights_quantization_method

    # Return the appropriate quantization parameters based on the quantization method
    if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                               QuantizationMethod.SYMMETRIC]:
        return {qi_inferable_quantizers_constants.NUM_BITS: attr_node_qc.weights_n_bits,
                qi_inferable_quantizers_constants.THRESHOLD: attr_node_qc.weights_quantization_params[THRESHOLD].flatten().tolist(),
                qi_inferable_quantizers_constants.PER_CHANNEL: attr_node_qc.weights_per_channel_threshold,
                qi_inferable_quantizers_constants.CHANNEL_AXIS: attr_node_qc.weights_channels_axis[0],  # output channel axis
                }

    elif quantization_method in [QuantizationMethod.UNIFORM]:
        return {qi_inferable_quantizers_constants.NUM_BITS: attr_node_qc.weights_n_bits,
                qi_inferable_quantizers_constants.PER_CHANNEL: attr_node_qc.weights_per_channel_threshold,
                qi_inferable_quantizers_constants.MIN_RANGE: attr_node_qc.weights_quantization_params[RANGE_MIN].flatten().tolist(),
                qi_inferable_quantizers_constants.MAX_RANGE: attr_node_qc.weights_quantization_params[RANGE_MAX].flatten().tolist(),
                qi_inferable_quantizers_constants.CHANNEL_AXIS: attr_node_qc.weights_channels_axis[0],  # output channel axis
                }

    elif quantization_method in [QuantizationMethod.LUT_POT_QUANTIZER, QuantizationMethod.LUT_SYM_QUANTIZER]:
        return {qi_inferable_quantizers_constants.NUM_BITS: attr_node_qc.weights_n_bits,
                qi_inferable_quantizers_constants.LUT_VALUES: attr_node_qc.weights_quantization_params[LUT_VALUES].flatten().tolist(),
                qi_inferable_quantizers_constants.THRESHOLD: attr_node_qc.weights_quantization_params[SCALE_PER_CHANNEL].flatten().tolist(),
                qi_inferable_quantizers_constants.PER_CHANNEL: attr_node_qc.weights_per_channel_threshold,
                qi_inferable_quantizers_constants.CHANNEL_AXIS: attr_node_qc.weights_channels_axis[0],  # output channel axis
                qi_inferable_quantizers_constants.INPUT_RANK: len(attr_node_qc.weights_quantization_params[SCALE_PER_CHANNEL].shape)}
                # TODO: Add LUT_VALUES_BITWIDTH & EPS to node quantization config

    else:
        Logger.critical(f'Not supported quantization method for weights inferable quantizers.')  # pragma: no cover


def get_activation_inferable_quantizer_kwargs(node_qc: NodeActivationQuantizationConfig) -> Dict[str, Any]:
    """
    Get the quantization parameters for an activation inferable quantizer.

    Args:
        node_qc: The node quantization configuration of the node for which the quantizer is being created.
            Needs to match the specific quantization target.

    Returns:
        The quantization parameters as a dictionary.
    """

    if not isinstance(node_qc, NodeActivationQuantizationConfig):
        Logger.critical(
            f"Non-compatible node quantization config was given for quantization target Activation.")  # pragma: no cover

    quantization_method = node_qc.activation_quantization_method

    # Return the appropriate quantization parameters based on the quantization method
    if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                               QuantizationMethod.SYMMETRIC]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.THRESHOLD: [node_qc.activation_quantization_params[THRESHOLD]],
                qi_inferable_quantizers_constants.SIGNED: node_qc.activation_quantization_params.get(SIGNED)}

    elif quantization_method in [QuantizationMethod.UNIFORM]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.MIN_RANGE: [node_qc.activation_quantization_params[RANGE_MIN]],
                qi_inferable_quantizers_constants.MAX_RANGE: [node_qc.activation_quantization_params[RANGE_MAX]]}

    elif quantization_method in [QuantizationMethod.LUT_POT_QUANTIZER]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.LUT_VALUES: node_qc.activation_quantization_params[LUT_VALUES].flatten().tolist(),
                qi_inferable_quantizers_constants.THRESHOLD: [node_qc.activation_quantization_params[THRESHOLD]],
                qi_inferable_quantizers_constants.SIGNED: node_qc.activation_quantization_params.get(SIGNED)}
        # TODO: Add LUT_VALUES_BITWIDTH & EPS to node quantization config
    else:
        Logger.critical(f'Not supported quantization method for inferable quantizers.')  # pragma: no cover


def get_weights_quantizer_for_node(node: BaseNode, attr_name: str) -> BasePyTorchInferableQuantizer:
    """
    Get weights quantizer for a node.

    Args:
        node: Node to create a weight quantizer for.
        attr_name: Attribute name to get its quantizer.

    Returns:
        Quantizer for the node's weights.

    """
    if node.final_weights_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final weights quantization configuration')  # pragma:
        # no cover
    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.get_attr_config(attr_name).weights_quantization_method

    quantier_for_node = get_inferable_quantizer_class(QuantizationTarget.Weights,
                                                      weights_quantization_method,
                                                      BasePyTorchInferableQuantizer)
    kwargs = get_weights_inferable_quantizer_kwargs(node_w_qc, attr_name)

    return quantier_for_node(**kwargs)


def get_activations_quantizer_for_node(node: BaseNode) -> BasePyTorchInferableQuantizer:
    """
    Get activation quantizer for a node.

    Args:
        node: Node to create an activation quantizer for.

    Returns:
        Quantizer for the node's activations.

    """
    if node.final_activation_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final activation quantization configuration')  #
        # pragma: no cover
    node_act_qc = node.final_activation_quantization_cfg
    activation_quantization_method = node_act_qc.activation_quantization_method

    quantizer_for_node = get_inferable_quantizer_class(QuantizationTarget.Activation,
                                                       activation_quantization_method,
                                                       BasePyTorchInferableQuantizer)
    kwargs = get_activation_inferable_quantizer_kwargs(node_act_qc)

    return quantizer_for_node(**kwargs)

