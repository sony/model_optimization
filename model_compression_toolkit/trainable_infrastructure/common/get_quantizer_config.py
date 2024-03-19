# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig, TrainableQuantizerCandidateConfig


def get_trainable_quantizer_weights_config(
        n: BaseNode,
        attr_name: str,
        weights_quantization_candidates: List[TrainableQuantizerCandidateConfig] = None
) -> TrainableQuantizerWeightsConfig:
    """
    Returns the relevant configuration for weights trainable quantizer

    Args:
        n: BaseNode - the node to build a trainable quantizer from.
        attr_name: Attribute name to get its weights quantizer configuration.
        weights_quantization_candidates: A list of weights quantizer config candidates.

    Returns:
         TrainableQuantizerWeightsConfig: an object that contains the quantizer configuration
    """
    if n.final_weights_quantization_cfg is None:
        Logger.critical(
            "The node requires a 'final_weights_quantization_cfg' configuration to build a "
            "quantizer. Please ensure this configuration is set for the node.")# pragma: no cover

    final_node_cfg = n.final_weights_quantization_cfg
    final_attr_cfg = final_node_cfg.get_attr_config(attr_name)
    return TrainableQuantizerWeightsConfig(final_attr_cfg.weights_quantization_method,
                                           final_attr_cfg.weights_n_bits,
                                           final_attr_cfg.weights_quantization_params,
                                           final_attr_cfg.enable_weights_quantization,
                                           final_attr_cfg.weights_channels_axis[0],  # Output channel axis
                                           final_attr_cfg.weights_per_channel_threshold,
                                           final_node_cfg.min_threshold,
                                           weights_quantization_candidates)


def get_trainable_quantizer_activation_config(
        n: BaseNode,
        activation_quantization_candidates: List[TrainableQuantizerCandidateConfig] = None
) -> TrainableQuantizerActivationConfig:
    """
    Returns configuration for activation trainable quantizer

    Args:
        n: BaseNode - the node to build a trainable quantizer from.
        activation_quantization_candidates: A list of activation quantizer candidates config.

    Returns:
         TrainableQuantizerActivationConfig - an object that contains the quantizer configuration
    """
    if n.final_activation_quantization_cfg is None:
        Logger.critical(
            "The node requires a 'final_activation_quantization_cfg' configuration to build a "
            "quantizer. Please ensure this configuration is set for the node.")# pragma: no cover

    final_cfg = n.final_activation_quantization_cfg
    return TrainableQuantizerActivationConfig(final_cfg.activation_quantization_method,
                                              final_cfg.activation_n_bits,
                                              final_cfg.activation_quantization_params,
                                              final_cfg.enable_activation_quantization,
                                              final_cfg.min_threshold,
                                              activation_quantization_candidates)


def get_trainable_quantizer_quantization_candidates(n: BaseNode, attr: str = None):
    """
    Returns quantization configuration candidates for activation and weights trainable quantizer.
    Checks that the candidates are compatible with trainable quantizer

    Args:
        n: BaseNode - the node to build a trainable quantizer from
        attr: Weights attribute to get its quantization configuration candidates and trainable quantizer.

    Returns:
         weights_quantization_candidates - A list of configuration candidates for weights
         activation_quantization_candidates - A list of configuration candidates for activation
    """

    if attr is not None:
        # all candidates must have the same weights quantization method
        weights_quantization_methods = set([cfg.weights_quantization_cfg.get_attr_config(attr).weights_quantization_method
             for cfg in n.candidates_quantization_cfg])
        if len(weights_quantization_methods) > 1:
            Logger.critical(f"Invalid 'candidates_quantization_cfg': Inconsistent weights "
                            f"quantization methods detected: {weights_quantization_methods}. "
                            f"Trainable quantizer requires all candidates to have the same weights "
                            f"quantization method.")  # pragma: no cover

    # all candidates must have the same activation quantization method
    activation_quantization_methods = set([cfg.activation_quantization_cfg.activation_quantization_method
                                           for cfg in n.candidates_quantization_cfg])
    if len(activation_quantization_methods) > 1:
        Logger.critical(f"Invalid 'candidates_quantization_cfg': Inconsistent activation quantization "
                        f"methods detected: {activation_quantization_methods}. "
                        f"Trainable quantizer requires all candidates to have the same activation quantization method.")# pragma: no cover

    # get unique lists of candidates
    unique_weights_candidates = n.get_unique_weights_candidates(attr)
    unique_activation_candidates = n.get_unique_activation_candidates()

    # generate list of weights quantizer candidates
    weights_cfg_candidates = [TrainableQuantizerCandidateConfig(
        cfg.weights_quantization_cfg.get_attr_config(attr).weights_n_bits,
        cfg.weights_quantization_cfg.get_attr_config(attr).weights_quantization_params)
        for cfg in unique_weights_candidates]

    # generate list of activation quantizer candidates
    activation_cfg_candidates = [TrainableQuantizerCandidateConfig(
        cfg.activation_quantization_cfg.activation_n_bits,
        cfg.activation_quantization_cfg.activation_quantization_params) for cfg in unique_activation_candidates]

    return weights_cfg_candidates, activation_cfg_candidates
