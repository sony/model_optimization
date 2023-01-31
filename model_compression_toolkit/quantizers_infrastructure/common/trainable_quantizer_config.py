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
from abc import ABC
from typing import Dict, List
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class TrainableQuantizerCandidateConfig:

    def __init__(self,
                 n_bits: int,
                 quantization_params: Dict,
                 ):
        """
        Class for representing candidates of quantization configurations for trainable quantizer.
        It can be used for weights and activation quantization configuration.

        Args:
            n_bits (int): Number of bits to use for quantization.
            quantization_params (Dict): Dictionary that contains quantization params.
        """

        self.n_bits = n_bits
        self.quantization_params = quantization_params


class TrainableQuantizerActivationConfig:

    def __init__(self,
                 activation_quantization_method: QuantizationMethod,
                 activation_n_bits: int,
                 activation_quantization_params: Dict,
                 enable_activation_quantization: bool,
                 min_threshold: float,
                 activation_quantization_candidates: List[TrainableQuantizerCandidateConfig] = None,
                 ):
        """
        Attributes for configuring activations trainable quantizer.

        Args:
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            activation_n_bits (int): Number of bits to quantize the activations.
            activation_quantization_params (Dict): Dictionary that contains activation quantization params.
            enable_activation_quantization (bool): Whether to quantize the layer's activations or not.
            min_threshold (float): Minimum threshold to use during thresholds selection.
        """
        self.activation_quantization_method = activation_quantization_method
        self.activation_n_bits = activation_n_bits
        self.activation_quantization_params = activation_quantization_params
        self.enable_activation_quantization = enable_activation_quantization
        self.min_threshold = min_threshold
        self.activation_bits_candidates = activation_quantization_candidates


class TrainableQuantizerWeightsConfig:
    def __init__(self,
                 weights_quantization_method: QuantizationMethod,
                 weights_n_bits: int,
                 weights_quantization_params: Dict,
                 enable_weights_quantization: bool,
                 weights_channels_axis: int,
                 weights_per_channel_threshold: bool,
                 min_threshold: float,
                 weights_quantization_candidates: List[TrainableQuantizerCandidateConfig] = None,
                 ):
        """
        Attributes for configuring weights trainable quantizer.

        Args:
            weights_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for weights quantization.
            weights_n_bits (int): Number of bits to quantize the coefficients.
            weights_quantization_params (Dict): Dictionary that contains weights quantization params.
            enable_weights_quantization (bool): Whether to quantize the layer's weights or not.
            weights_channels_axis (int): Axis to quantize a node's kernel when quantizing per-channel.
            weights_per_channel_threshold (bool): Whether to quantize the weights per-channel or not (per-tensor).
            min_threshold (float): Minimum threshold to use during thresholds selection.
        """
        self.weights_quantization_method = weights_quantization_method
        self.weights_n_bits = weights_n_bits
        self.weights_quantization_params = weights_quantization_params
        self.enable_weights_quantization = enable_weights_quantization
        self.weights_channels_axis = weights_channels_axis
        self.weights_per_channel_threshold = weights_per_channel_threshold
        self.min_threshold = min_threshold
        self.weights_bits_candidates = weights_quantization_candidates
