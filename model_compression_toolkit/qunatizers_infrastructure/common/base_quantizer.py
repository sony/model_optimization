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

from typing import List
from enum import Enum

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig, BaseNodeQuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class QuantizationTarget(Enum):
    Activation = 0
    Weights = 1


class BaseQuantizer:
    def __init__(self,
                 quantization_config: BaseNodeQuantizationConfig,
                 quantization_target: QuantizationTarget,
                 quantization_method: List[QuantizationMethod]):
        """
        This class is a base quantizer which validate the the provide quantization config and define abstract function which any quantizer need to implment.

        Args:
            quantization_config: node quantization config class contins all the information above a quantizer.
            quantization_target: A enum which decided the qunaizer tensor type activation or weights.
            quantization_method: A list of "QuantizationMethod" enums which represent the quantizer supported methods.
        """
        self.quantization_config = quantization_config
        self.quantization_target = quantization_target
        self.quantization_method = quantization_method
        if self.quantization_target == QuantizationTarget.Weights:
            self.validate_weights()
            if self.quantization_config.weights_quantization_method not in quantization_method:
                common.Logger.error(
                    f'Quantization method mismatch expected:{quantization_method} and got  {self.quantization_config.weights_quantization_method}')
        elif self.quantization_target == QuantizationTarget.Activation:
            self.validate_activation()
            if self.quantization_config.activation_quantization_method not in quantization_method:
                common.Logger.error(
                    f'Quantization method mismatch expected:{quantization_method} and got  {self.quantization_config.activation_quantization_method}')
        else:
            common.Logger.error(
                f'Unknown Quantization Part:{quantization_target}')

    def initialize_quantization(self,
                                tensor_shape,
                                name: str,
                                layer):
        """
        This initializes the quantizer parameters given the parameter name and shape.

        Args:
            tensor_shape:  tensor shape
            name: tensor name
            layer: layer to quantized

        Returns: None

        """
        raise NotImplemented

    def __call__(self,
                 input2quantize,
                 training: bool):
        """
        Quantize a tensor.

        Args:
            input2quantize: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """
        raise NotImplemented

    def activation_quantization(self) -> bool:
        """

        Returns: A boolean stating is this activation quantizer

        """
        return isinstance(self.quantization_config, NodeActivationQuantizationConfig)

    def weights_quantization(self) -> bool:
        """

        Returns: A boolean stating is this weights quantizer

        """
        return isinstance(self.quantization_config, NodeWeightsQuantizationConfig)

    def validate_weights(self) -> None:
        """
        This function valid the quantize config compare with it parameters.


        """
        if self.activation_quantization() or not self.weights_quantization():
            common.Logger.error(f'Expect weight quantization got activation')

    def validate_activation(self) -> None:
        """
        This function valid the quantize config compare with it parameters.

        """
        if not self.activation_quantization() or self.weights_quantization():
            common.Logger.error(f'Expect activation quantization got weight')
