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
import numpy as np

from typing import List, Callable

from model_compression_toolkit.core.common import BaseNode, Logger
from model_compression_toolkit.core.common.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import calculate_delta
from model_compression_toolkit.core.common.target_platform import QuantizationMethod

# Supporting other quantizer types in the future
from model_compression_toolkit.exporter.fully_quantized.pytorch.quantizers.fq_quantizer import FakeQuantQuantizer
from model_compression_toolkit.exporter.fully_quantized.pytorch.quantizers.uniform_weights_quantizer import \
    UniformWeightsQuantizer
import torch

SUPPORTED_WEIGHT_QUANTIZER_TYPES = [QuantizationMethod.POWER_OF_TWO,
                                    QuantizationMethod.SYMMETRIC,
                                    QuantizationMethod.UNIFORM]

SUPPORTED_ACTIVATION_QUANTIZER_TYPES = [QuantizationMethod.POWER_OF_TWO,
                                        QuantizationMethod.SYMMETRIC,
                                        QuantizationMethod.UNIFORM]


def get_weights_quantizer_for_node(node: BaseNode) -> List[Callable]:
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

    if weights_quantization_method not in SUPPORTED_WEIGHT_QUANTIZER_TYPES:
        Logger.error(f'Fully quantized models are now supported for {SUPPORTED_WEIGHT_QUANTIZER_TYPES} quantization methods, but node has {weights_quantization_method} quantization method')

    # Compute quantizer params based on node's quantization params
    if weights_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
        weight_thresholds = node_w_qc.weights_quantization_params.get(THRESHOLD)
        assert weight_thresholds is not None
        if weights_quantization_method == QuantizationMethod.POWER_OF_TWO:
            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in weight_thresholds.flatten()])
            if not is_threshold_pot:
                Logger.error(f'Expected threshold to be power of 2 but is {weight_thresholds}')

        min_range = -weight_thresholds
        max_range = weight_thresholds - calculate_delta(weight_thresholds,
                                                        n_bits=node_w_qc.weights_n_bits,
                                                        signed=True)

    else:
        Logger.error(f'For now fully quantized models support only {SUPPORTED_WEIGHT_QUANTIZER_TYPES} for weights quantization, but found {weights_quantization_method}')

    return [UniformWeightsQuantizer(num_bits=node_w_qc.weights_n_bits,
                                    max_range=max_range,
                                    min_range=min_range,
                                    quantization_method=node_w_qc.weights_quantization_method,
                                    per_channel=node_w_qc.weights_per_channel_threshold,
                                    output_channels_axis=node_w_qc.weights_channels_axis
                                    )]


def get_activations_quantizer_for_node(node: BaseNode) -> List[Callable]:
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

    if activation_quantization_method not in SUPPORTED_ACTIVATION_QUANTIZER_TYPES:
        Logger.error(
            f'Fully quantized models are now supported for {SUPPORTED_ACTIVATION_QUANTIZER_TYPES} quantization methods, '
            f'but node has {activation_quantization_method} quantization method')

    activation_thresholds = node_act_qc.activation_quantization_params.get(THRESHOLD)

    if activation_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
        if activation_quantization_method == QuantizationMethod.POWER_OF_TWO:
            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in activation_thresholds.flatten()])
            if not is_threshold_pot:
                Logger.error(f'Expected threshold to be power of 2 but is {node_act_qc.activation_quantization_params.get(THRESHOLD)}')

        min_range = 0
        if node_act_qc.activation_quantization_params.get(SIGNED):
            min_range = -activation_thresholds

        max_range = activation_thresholds - calculate_delta(
            activation_thresholds,
            n_bits=node_act_qc.activation_n_bits,
            signed=node_act_qc.activation_quantization_params.get(SIGNED))

    elif activation_quantization_method in [QuantizationMethod.UNIFORM]:
        min_range = node_act_qc.activation_quantization_params.get(RANGE_MIN)
        max_range = node_act_qc.activation_quantization_params.get(RANGE_MAX)

    else:
        Logger.error(f'For now fully quantized models support only {SUPPORTED_ACTIVATION_QUANTIZER_TYPES} for activation quantization, but found {activation_quantization_method}')

    return [FakeQuantQuantizer(nbits=node_act_qc.activation_n_bits,
                              min_range=min_range,
                              max_range=max_range,
                              quantization_method=activation_quantization_method)]
