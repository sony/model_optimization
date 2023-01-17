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
from model_compression_toolkit.quantizers_infrastructure import pytorch_inferable_quantizers
from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers import constants as qi_inferable_quantizers_constants

QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER = {
    QuantizationMethod.POWER_OF_TWO: pytorch_inferable_quantizers.WeightsPOTInferableQuantizer,
    QuantizationMethod.SYMMETRIC: pytorch_inferable_quantizers.WeightsSymmetricInferableQuantizer,
    QuantizationMethod.UNIFORM: pytorch_inferable_quantizers.WeightsUniformInferableQuantizer
}

QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER = {
    QuantizationMethod.POWER_OF_TWO: pytorch_inferable_quantizers.ActivationPOTInferableQuantizer,
    QuantizationMethod.SYMMETRIC: pytorch_inferable_quantizers.ActivationSymmetricInferableQuantizer,
    QuantizationMethod.UNIFORM: pytorch_inferable_quantizers.ActivationUniformInferableQuantizer
}

def get_weights_inferable_quantizer_kwargs(node: BaseNode) -> Dict[str, Any]:
    # Get the weights quantization configuration for the node
    node_w_qc = node.final_weights_quantization_cfg
    quantization_method = node_w_qc.weights_quantization_method

    # Check if the quantization method is supported for inferable quantizers
    assert quantization_method in QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER, f'{quantization_method} for weights ' \
                                                                           f'not in supported quantization ' \
                                                                           f'methods for inferable quantizers'

    # Return the appropriate quantization parameters based on the quantization method
    if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                               QuantizationMethod.SYMMETRIC]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_w_qc.weights_n_bits,
                qi_inferable_quantizers_constants.THRESHOLD: node_w_qc.weights_quantization_params.get(THRESHOLD),
                qi_inferable_quantizers_constants.SIGNED: True,
                qi_inferable_quantizers_constants.PER_CHANNEL: node_w_qc.weights_per_channel_threshold}

    elif quantization_method in [QuantizationMethod.UNIFORM]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_w_qc.weights_n_bits,
                qi_inferable_quantizers_constants.PER_CHANNEL: node_w_qc.weights_per_channel_threshold,
                qi_inferable_quantizers_constants.MIN_RANGE: node_w_qc.weights_quantization_params.get(RANGE_MIN),
                qi_inferable_quantizers_constants.MAX_RANGE: node_w_qc.weights_quantization_params.get(RANGE_MAX)}
    else:
        Logger.critical(f'Not supported quantization method for weights inferable quantizers.')


def get_activation_inferable_quantizer_kwargs(node: BaseNode) -> Dict[str, Any]:
    # Get the activation quantization configuration for the node
    node_qc = node.final_activation_quantization_cfg
    quantization_method = node_qc.activation_quantization_method

    # Check if the quantization method is supported for inferable quantizers
    assert quantization_method in QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER, f'{quantization_method} for weights ' \
                                                                              f'not in ' \
                                                                              f'supported quantization methods ' \
                                                                              f'for inferable' \
                                                                              f' quantizers'

    # Return the appropriate quantization parameters based on the quantization method
    if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                               QuantizationMethod.SYMMETRIC]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.THRESHOLD: node_qc.activation_quantization_params.get(THRESHOLD),
                qi_inferable_quantizers_constants.SIGNED: node_qc.activation_quantization_params.get(SIGNED)}

    elif quantization_method in [QuantizationMethod.UNIFORM]:
        return {qi_inferable_quantizers_constants.NUM_BITS: node_qc.activation_n_bits,
                qi_inferable_quantizers_constants.MIN_RANGE: node_qc.activation_quantization_params.get(RANGE_MIN),
                qi_inferable_quantizers_constants.MAX_RANGE: node_qc.activation_quantization_params.get(RANGE_MAX)}
    else:
        Logger.critical(f'Not supported quantization method for inferable quantizers.')


def get_weights_quantizer_for_node(node: BaseNode) -> pytorch_inferable_quantizers.BasePyTorchInferableQuantizer:
    """
    Get weights quantizer for a node.

    Args:
        node: Node to create a weight quantizer for.

    Returns:
        Quantizer for the node's weights.

    """
    if node.final_weights_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final weights quantization configuration')
    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.weights_quantization_method
    kwargs = get_weights_inferable_quantizer_kwargs(node)
    return QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER.get(weights_quantization_method)(**kwargs)


def get_activations_quantizer_for_node(node: BaseNode) -> pytorch_inferable_quantizers.BasePyTorchInferableQuantizer:
    """
    Get activation quantizer for a node.

    Args:
        node: Node to create an activation quantizer for.

    Returns:
        Quantizer for the node's activations.

    """
    if node.final_activation_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final activation quantization configuration')
    node_act_qc = node.final_activation_quantization_cfg
    activation_quantization_method = node_act_qc.activation_quantization_method
    kwargs = get_activation_inferable_quantizer_kwargs(node)
    return QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER.get(activation_quantization_method)(**kwargs)

