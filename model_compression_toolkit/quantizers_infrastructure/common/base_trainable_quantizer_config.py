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

from typing import List, Dict
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class BaseQuantizerConfig(object):
    """
    Base class for quantizer configuration
    """
    def __init__(self):
        return


class TrainableQuantizerActivationConfig(BaseQuantizerConfig):

    def __init__(self,
                 activation_quantization_method: QuantizationMethod,
                 activation_n_bits: int,
                 activation_quantization_params: Dict,
                 enable_activation_quantization: bool,
                 min_threshold: float,
                 ):
        """
        Attributes for configuring activations trainable quantizer.

        Args:
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            activation_n_bits (int): Number of bits to quantize the activations.
            activation_quantization_params (Dict): Dictionary that contains activation quantization params.
            enable_activation_quantization (bool): Whether to quantize the model activations or not.
            min_threshold (float): Minimum threshold to use during thresholds selection.
        """
        self.activation_quantization_method = activation_quantization_method
        self.activation_n_bits = activation_n_bits
        self.activation_quantization_params = activation_quantization_params
        self.enable_activation_quantization = enable_activation_quantization
        self.min_threshold = min_threshold

    def set_activation_quantization_param(self,
                                          activation_params: dict):
        """
         Set a quantization parameter for the activation quantizer.

        Args:
            activation_params: Dictionary that contains activation quantization params.

        """
        assert self.enable_activation_quantization
        for param_name, param_value in activation_params.items():
            self.activation_quantization_params[param_name] = param_value

    def has_activation_quantization_params(self) -> bool:
        """

        Returns: Whether ActivationTrainableQuantizerConfig has activation quantization params or not.

        """
        return len(self.activation_quantization_params) > 0


class TrainableQuantizerWeightsConfig(BaseQuantizerConfig):
    def __init__(self,
                 weights_quantization_method: QuantizationMethod,
                 weights_n_bits: int,
                 weights_quantization_params: Dict,
                 enable_weights_quantization: bool,
                 weights_channels_axis: int,
                 weights_per_channel_threshold: bool,
                 min_threshold: float,
                 ):
        """
        Attributes for configuring weights trainable quantizer.

        Args:
            weights_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for weights quantization.
            weights_n_bits (int): Number of bits to quantize the coefficients.
            weights_quantization_params (Dict): Dictionary that contains weights quantization params.
            enable_weights_quantization (bool): Whether to quantize the model weights or not.
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

    def set_weights_quantization_param(self,
                                       weights_params: dict):
        """
         Set a quantization parameter for the weights quantizer.

        Args:
            weights_params: Dictionary that contains weight quantization params.

        """
        assert self.enable_weights_quantization
        for param_name, param_value in weights_params.items():
            self.weights_quantization_params[param_name] = param_value

    def has_weights_quantization_params(self) -> bool:
        """

        Returns: Whether WeightsTrainableQuantizerConfig has weights quantization params or not.

        """
        return len(self.weights_quantization_params) > 0

