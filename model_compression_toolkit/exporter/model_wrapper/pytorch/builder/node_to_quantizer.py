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

from model_compression_toolkit.core.common import BaseNode, Logger
from model_compression_toolkit.core.common.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.exporter.model_wrapper.common.exporter_get_quantizer import get_quantizer_class
from model_compression_toolkit.quantizers_infrastructure import pytorch_inferable_quantizers, QuantizationTarget, \
    BasePyTorchInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers import constants as qi_inferable_quantizers_constants
import numpy as np


def get_weights_inferable_quantizer_kwargs(node: BaseNode) -> Dict[str, Any]:
    # Get the weights quantization configuration for the node
    node_w_qc = node.final_weights_quantization_cfg
    quantization_method = node_w_qc.weights_quantization_method

    # Return the appropriate quantization parameters based on the quantization method
    if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                               QuantizationMethod.SYMMETRIC]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_w_qc.weights_n_bits,
                qi_inferable_quantizers_constants.THRESHOLD: node_w_qc.weights_quantization_params[THRESHOLD].flatten(),
                qi_inferable_quantizers_constants.PER_CHANNEL: node_w_qc.weights_per_channel_threshold,
                qi_inferable_quantizers_constants.CHANNEL_AXIS: node_w_qc.weights_channels_axis}

    elif quantization_method in [QuantizationMethod.UNIFORM]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_w_qc.weights_n_bits,
                qi_inferable_quantizers_constants.PER_CHANNEL: node_w_qc.weights_per_channel_threshold,
                qi_inferable_quantizers_constants.MIN_RANGE: node_w_qc.weights_quantization_params[RANGE_MIN].flatten(),
                qi_inferable_quantizers_constants.MAX_RANGE: node_w_qc.weights_quantization_params[RANGE_MAX].flatten(),
                qi_inferable_quantizers_constants.CHANNEL_AXIS: node_w_qc.weights_channels_axis}
    else:
        Logger.critical(f'Not supported quantization method for weights inferable quantizers.')  # pragma: no cover


def get_activation_inferable_quantizer_kwargs(node: BaseNode) -> Dict[str, Any]:
    # Get the activation quantization configuration for the node
    node_qc = node.final_activation_quantization_cfg
    quantization_method = node_qc.activation_quantization_method

    # Return the appropriate quantization parameters based on the quantization method
    if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                               QuantizationMethod.SYMMETRIC]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.THRESHOLD: np.asarray([node_qc.activation_quantization_params[THRESHOLD]]),
                qi_inferable_quantizers_constants.SIGNED: node_qc.activation_quantization_params.get(SIGNED)}

    elif quantization_method in [QuantizationMethod.UNIFORM]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.MIN_RANGE: np.asarray([node_qc.activation_quantization_params[RANGE_MIN]]),
                qi_inferable_quantizers_constants.MAX_RANGE: np.asarray([node_qc.activation_quantization_params[RANGE_MAX]])}
    else:
        Logger.critical(f'Not supported quantization method for inferable quantizers.')  # pragma: no cover


def get_weights_quantizer_for_node(node: BaseNode) -> pytorch_inferable_quantizers.BasePyTorchInferableQuantizer:
    """
    Get weights quantizer for a node.

    Args:
        node: Node to create a weight quantizer for.

    Returns:
        Quantizer for the node's weights.

    """
    if node.final_weights_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final weights quantization configuration')  # pragma:
        # no cover
    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.weights_quantization_method

    quantier_for_node = get_quantizer_class(QuantizationTarget.Weights,
                                            weights_quantization_method,
                                            BasePyTorchInferableQuantizer)
    kwargs = get_weights_inferable_quantizer_kwargs(node)

    return quantier_for_node(**kwargs)


def get_activations_quantizer_for_node(node: BaseNode) -> pytorch_inferable_quantizers.BasePyTorchInferableQuantizer:
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

    quantier_for_node = get_quantizer_class(QuantizationTarget.Activation,
                                            activation_quantization_method,
                                            BasePyTorchInferableQuantizer)
    kwargs = get_activation_inferable_quantizer_kwargs(node)

    return quantier_for_node(**kwargs)

